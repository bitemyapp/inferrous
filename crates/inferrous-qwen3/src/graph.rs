//! CUDA graph ownership for the fixed, stream-serialized decode workspace.
use anyhow::Result;
use cuda_core::{CudaStream, IntoResult, sys};
use std::{ptr, sync::Arc};

struct SourceGraph {
    raw: sys::CUgraph,
    stream: Arc<CudaStream>,
}
impl Drop for SourceGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let ctx = self.stream.context();
            ctx.record_err(ctx.bind_to_thread());
            // SAFETY: sole owner of the completed capture; instantiation copies it.
            ctx.record_err(unsafe { sys::cuGraphDestroy(self.raw).result() });
        }
    }
}
struct Capture {
    stream: Arc<CudaStream>,
    active: bool,
}
impl Capture {
    fn end(mut self) -> Result<SourceGraph> {
        let mut graph = SourceGraph {
            raw: ptr::null_mut(),
            stream: self.stream.clone(),
        };
        // End capture even if enqueueing a node failed; an invalidated capture
        // returns an error and a null graph, with the stream no longer capturing.
        let result =
            unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph.raw).result() };
        self.active = false;
        result?;
        Ok(graph)
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            // Also unwind a capture when host code panics while recording it.
            let mut graph = SourceGraph {
                raw: ptr::null_mut(),
                stream: self.stream.clone(),
            };
            let ctx = self.stream.context();
            ctx.record_err(ctx.bind_to_thread());
            ctx.record_err(unsafe {
                sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph.raw).result()
            });
        }
    }
}

/// Turn every captured kernel-to-kernel edge into a programmatic edge, so a
/// kernel launches once all predecessor CTAs pass `griddepcontrol.wait` and
/// admit it. Device code waits for full predecessor completion before touching
/// any produced, consumed or reused buffer; see kernels::dependency_wait.
///
/// # Safety
/// `graph` is a valid, uninstantiated graph whose kernel nodes all wait.
unsafe fn programmatic_edges(graph: sys::CUgraph) -> Result<()> {
    let mut count = 0;
    unsafe {
        sys::cuGraphGetEdges_v2(
            graph,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut count,
        )
        .result()?;
    }
    let mut from = vec![ptr::null_mut(); count];
    let mut to = vec![ptr::null_mut(); count];
    // SAFETY: CUgraphEdgeData is plain bytes; zero is the default edge.
    let mut data = vec![unsafe { std::mem::zeroed::<sys::CUgraphEdgeData>() }; count];
    unsafe {
        sys::cuGraphGetEdges_v2(
            graph,
            from.as_mut_ptr(),
            to.as_mut_ptr(),
            data.as_mut_ptr(),
            &mut count,
        )
        .result()?;
    }
    let kernel = |node| -> Result<bool> {
        let mut kind = 0;
        unsafe { sys::cuGraphNodeGetType(node, &mut kind).result()? };
        Ok(kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL)
    };
    let mut edges = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..count {
        let default = data[i].type_ == 0 && data[i].from_port == 0;
        if default && kernel(from[i])? && kernel(to[i])? {
            edges.0.push(from[i]);
            edges.1.push(to[i]);
            edges.2.push(data[i]);
        }
    }
    let mut programmatic = unsafe { std::mem::zeroed::<sys::CUgraphEdgeData>() };
    programmatic.from_port = sys::CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC as u8;
    programmatic.type_ =
        sys::CUgraphDependencyType_enum_CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC as u8;
    let replacement = vec![programmatic; edges.0.len()];
    unsafe {
        sys::cuGraphRemoveDependencies_v2(
            graph,
            edges.0.as_ptr(),
            edges.1.as_ptr(),
            edges.2.as_ptr(),
            edges.0.len(),
        )
        .result()?;
        sys::cuGraphAddDependencies_v2(
            graph,
            edges.0.as_ptr(),
            edges.1.as_ptr(),
            replacement.as_ptr(),
            edges.0.len(),
        )
        .result()?;
    }
    Ok(())
}

pub(super) struct Graph {
    raw: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}
// SAFETY: CUDA graph handles are not thread-local. Every operation binds the
// owning context; the enclosing execution mutex excludes concurrent use.
unsafe impl Send for Graph {}
impl Graph {
    /// Capture kernels whose module and device allocations outlive this graph.
    ///
    /// # Safety
    /// The caller must retain every referenced allocation/module, synchronize
    /// before mutating them outside the stream, and serialize all launches.
    pub(super) unsafe fn capture(
        stream: &Arc<CudaStream>,
        programmatic: bool,
        record: impl FnOnce() -> Result<()>,
    ) -> Result<Self> {
        stream.context().bind_to_thread()?;
        unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .result()?;
        }
        let capture = Capture {
            stream: stream.clone(),
            active: true,
        };
        let recorded = record();
        let source = capture.end();
        recorded?;
        let source = source?;
        let mut graph = Self {
            raw: ptr::null_mut(),
            stream: stream.clone(),
        };
        unsafe {
            if programmatic {
                programmatic_edges(source.raw)?;
            }
            sys::cuGraphInstantiateWithFlags(&mut graph.raw, source.raw, 0).result()?;
        }
        Ok(graph)
    }
    /// # Safety
    /// All captured allocations remain valid and are exclusively owned by the
    /// caller until the stream completes. Control updates precede this launch.
    pub(super) unsafe fn launch(&self) -> Result<()> {
        self.stream.context().bind_to_thread()?;
        unsafe {
            sys::cuGraphLaunch(self.raw, self.stream.cu_stream()).result()?;
        }
        Ok(())
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let ctx = self.stream.context();
            // Complete any failed request's queued nodes before its buffers drop.
            ctx.record_err(self.stream.synchronize());
            ctx.record_err(unsafe { sys::cuGraphExecDestroy(self.raw).result() });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_capture_releases_stream() -> Result<()> {
        let ctx = cuda_core::CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        // SAFETY: the failed capture references no allocations or modules.
        let result = unsafe {
            Graph::capture(&stream, true, || {
                anyhow::bail!("injected recording failure")
            })
        };
        assert!(result.is_err());
        let data = cuda_core::DeviceBuffer::from_host(&stream, &[17_u32])?;
        assert_eq!(data.to_host_vec(&stream)?, [17]);
        Ok(())
    }
}
