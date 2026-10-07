//! Immutable driver-encoded TMA maps for the model's stable device allocations.
use anyhow::Result;
use cuda_core::{CudaStream, DeviceBuffer, IntoResult, sys};

pub(crate) fn map(
    stream: &CudaStream,
    source: &DeviceBuffer<i8>,
    rows: usize,
    width: usize,
    tile_rows: u32,
) -> Result<DeviceBuffer<u64>> {
    anyhow::ensure!(
        rows > 0 && width.is_multiple_of(128) && source.len() == rows * width,
        "invalid TMA source shape"
    );
    anyhow::ensure!(source.cu_deviceptr() % 128 == 0, "unaligned TMA source");
    stream.context().bind_to_thread()?;
    let mut descriptor = sys::CUtensorMap { opaque: [0; 16] };
    // SAFETY: aligned descriptor and allocation, exact dimensions/strides,
    // fixed 128-byte inner tile, and no host mutation of the encoded descriptor.
    unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut descriptor,
            sys::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            source.cu_deviceptr() as *mut _,
            [width as u64, rows as u64].as_ptr(),
            [width as u64].as_ptr(),
            [128, tile_rows].as_ptr(),
            [1, 1].as_ptr(),
            sys::CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            sys::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            sys::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
        .result()?;
    }
    Ok(DeviceBuffer::from_host(stream, &descriptor.opaque)?)
}
