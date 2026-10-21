"""Execute a frozen batch comparison plan, preserving whole resident suites.

Without --execute, check input/image identities and report the planned work.
This runner does not certify model correctness, exhaustive stock eligibility or
performance acceptance. Those remain prerequisites and a separate final audit.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from compare_batch_performance import collect


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


def option(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f'expected exactly one {flag}')
    return argv[argv.index(flag) + 1]


def host_path(plan, value):
    root = Path(plan['cwd']).resolve()
    if value.startswith('/workspace/'):
        value = value.removeprefix('/workspace/')
    path = (root / value).resolve()
    path.relative_to(root)
    return path


def pinned_hash(plan, path):
    relative = str(path.relative_to(Path(plan['cwd']).resolve()))
    return plan['inputs_sha256'][relative]


def verify_inputs(plan):
    for name, expected in plan['inputs_sha256'].items():
        if sha(host_path(plan, name)) != expected:
            raise ValueError(f'pinned input changed: {name}')
    for name, image in plan['images'].items():
        actual = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image['reference']], text=True).strip()
        if actual != image['image_id']:
            raise ValueError(f'pinned image changed: {name}')


def fingerprint(plan, job):
    # Changes to another job's output path can recover a failed whole suite
    # without invalidating successful suites. Their own contracts stay frozen.
    context = {key: plan[key] for key in ['cwd', 'inputs_sha256', 'images', 'environment']}
    return hashlib.sha256(json.dumps(dict(context=context, job=job), sort_keys=True).encode()).hexdigest()


def validate_job(plan, job):
    argv = job['argv']
    if job['engine'] not in ['local', 'stock']:
        raise ValueError('unknown planned engine')
    local = job['engine'] == 'local'
    mode, prefix = {'cold-reuse': ('reuse', False), 'prefix-reuse': ('reuse', True),
                    'recompute': ('recompute', False)}[job['mode']]
    if option(argv, '--cache-mode') != mode:
        raise ValueError('command cache mode differs from planned mode')
    output = host_path(plan, job['output'])
    if output != host_path(plan, option(argv, '--output')):
        raise ValueError('command output differs from planned output')
    model = host_path(plan, option(argv, '--model'))
    inputs = [(model if local else model.parent) / 'manifest.json',
              host_path(plan, option(argv, '--workloads' if local else '--requests'))]
    if local:
        config = host_path(plan, option(argv, '--config'))
        inputs += [config, host_path(plan, option(argv, '--binary'))]
        settings = json.loads(config.read_text())
        if settings['verify_logits'] or (settings['prefix_cache_entries'] > 0) != prefix:
            raise ValueError('command configuration differs from planned prefix/verification policy')
    elif ('--prefix-cache' in argv) != prefix or argv[1] != job['candidate']['image']:
        raise ValueError('command prefix/image differs from planned policy')
    for path in inputs:
        pinned_hash(plan, path)
    if output in {host_path(plan, name) for name in plan['inputs_sha256']}:
        raise ValueError('output overlaps a pinned input')
    warmup, measured = job['warmups_per_workload'], job['measured_per_workload']
    if (type(warmup) is not int or type(measured) is not int or warmup < 0 or measured < 1
            or int(option(argv, '--repeat')) != warmup + measured
            or job['expected_records'] != plan['scenarios'] * (warmup + measured)):
        raise ValueError('planned sample counts differ from command')


def validate_suite(plan, job, device):
    argv = job['argv']
    local = job['engine'] == 'local'
    workpath = host_path(plan, option(argv, '--workloads' if local else '--requests'))
    if sha(workpath) != pinned_hash(plan, workpath):
        raise ValueError('workload identity changed')
    workloads = json.loads(workpath.read_text())
    output = host_path(plan, job['output'])
    if output != host_path(plan, option(argv, '--output')):
        raise ValueError('command output differs from planned output')
    with output.open() as stream:
        records = [json.loads(line) for line in stream]
    repetitions = job['warmups_per_workload'] + job['measured_per_workload']
    if (len(records) != job['expected_records'] or len(records) != len(workloads) * repetitions
            or int(option(argv, '--repeat')) != repetitions):
        raise ValueError('incomplete or mismatched suite record count')
    matched, series = collect([(str(output), records)], workloads, local,
                              job['warmups_per_workload'], job['measured_per_workload'])
    if len(series) != 1 or matched['device'] != device:
        raise ValueError('suite engine/device identity mismatch')
    first = records[0]
    model = host_path(plan, option(argv, '--model'))
    manifest = (model if local else model.parent) / 'manifest.json'
    if first['model_root'] != pinned_hash(plan, manifest) or first['cache_mode'] != option(argv, '--cache-mode'):
        raise ValueError('suite model/cache identity mismatch')
    if local:
        config = host_path(plan, option(argv, '--config'))
        binary = host_path(plan, option(argv, '--binary'))
        if (sha(config) != pinned_hash(plan, config) or first['engine_config'] != json.loads(config.read_text())
                or first['binary_sha256'] != pinned_hash(plan, binary)):
            raise ValueError('suite binary/configuration identity mismatch')
    else:
        candidate = job['candidate']
        image = plan['images'][candidate['image']]
        image_ref = image['image_id'] if candidate['image'] == 'vllm-b12x' else image['reference']
        expected = dict(engine='vllm', linear_backend=option(argv, '--linear-backend'),
                        attention_backend=option(argv, '--attention-backend'),
                        quantization=option(argv, '--quantization'), weight_scale_dtype='torch.float32',
                        hardware_profile=option(argv, '--hardware-profile'),
                        max_context=int(option(argv, '--max-context')), max_num_seqs=int(option(argv, '--max-num-seqs')),
                        kv_cache_bytes=int(option(argv, '--kv-cache-bytes')), prefix_cache='--prefix-cache' in argv,
                        graphs_without_compile='--graphs-without-compile' in argv,
                        disable_cutedsl_warmup='--disable-cutedsl-warmup' in argv, eager=False, graph_size_one=False)
        expected['memory_fraction'] = {'rtx5090': 0.70, 'rtx-pro-6000': 0.30}[expected['hardware_profile']]
        n = expected['max_num_seqs']
        expected['graph_capture_sizes'] = sorted({1 << power for power in range(n.bit_length())} | {n}) if expected['graphs_without_compile'] else None
        if (any(first.get(key) != value for key, value in expected.items())
                or first['environment'].get('INFERROUS_STOCK_IMAGE') != image_ref):
            raise ValueError('suite stock configuration/image identity mismatch')
    return dict(records=len(records), contract=matched, series_sha256=series[0]['series_sha256'])


def paths_for(output):
    return {name: output.with_suffix(suffix) for name, suffix in dict(
        started='.started.json', process='.process.json', result='.result.json',
        stdout='.stdout.log', stderr='.stderr.log', telemetry='.telemetry.csv',
        telemetry_stderr='.telemetry-stderr.log').items()}


def require_idle_gpu():
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU already has a compute process; leave it running and resume later')


def run_measured(plan, job, paths):
    environment = os.environ | plan['environment']
    with paths['telemetry'].open('x') as telemetry, paths['telemetry_stderr'].open('x') as monitor_error:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw',
                                    '--format=csv', '-lms', '500'], stdout=telemetry, stderr=monitor_error)
        try:
            with paths['stdout'].open('x') as out, paths['stderr'].open('x') as err:
                process = subprocess.Popen(job['argv'], cwd=plan['cwd'], env=environment,
                                           stdout=out, stderr=err, start_new_session=True)
                write_new(paths['process'], dict(pid=process.pid, started_unix=time.time(), argv=job['argv']))
                code = process.wait()
                if monitor.poll() is not None:
                    raise RuntimeError('telemetry monitor exited before the suite completed')
                return code
        finally:
            if monitor.poll() is None:
                monitor.terminate()
            monitor.wait()


def execute_suite(plan, job, device, executor_sha256):
    validate_job(plan, job)
    output = host_path(plan, job['output'])
    paths = paths_for(output)
    job_hash = fingerprint(plan, job)
    if paths['result'].exists():
        saved = json.loads(paths['result'].read_text())
        if (not saved['complete'] or saved['exit_code'] != 0 or saved['job_sha256'] != job_hash
                or saved['executor_sha256'] != executor_sha256 or saved['device'] != device):
            raise RuntimeError(f'{job["name"]}: failed or changed prior suite; preserve artifacts and use a fresh output path')
        expected_paths = {str(output)} | {str(path) for name, path in paths.items() if name != 'result'}
        if set(saved['artifact_sha256']) != expected_paths or any(sha(Path(path)) != digest for path, digest in saved['artifact_sha256'].items()):
            raise ValueError('completed suite artifacts changed')
        validate_suite(plan, job, device)
        return saved
    if output.exists() or any(path.exists() for path in paths.values()):
        raise RuntimeError(f'{job["name"]}: partial or active suite; refuse to overwrite or restart it')
    require_idle_gpu()
    write_new(paths['started'], dict(job_sha256=job_hash, executor_sha256=executor_sha256,
                                    executor_pid=os.getpid(), started_unix=time.time(), job=job))
    result = dict(job_sha256=job_hash, executor_sha256=executor_sha256, device=device,
                  complete=False, exit_code=None)
    try:
        result['exit_code'] = run_measured(plan, job, paths)
        if result['exit_code'] != 0:
            raise RuntimeError(f'suite process exited {result["exit_code"]}')
        verify_inputs(plan)
        validation = validate_suite(plan, job, device)
        with paths['telemetry'].open() as stream:
            samples = list(csv.reader(stream))
        if len(samples) < 2 or any(len(row) != 8 for row in samples):
            raise ValueError('missing or malformed GPU telemetry')
        result.update(complete=True, validation=validation, telemetry_samples=len(samples) - 1,
                      artifact_sha256={str(path): sha(path) for path in [output] + [p for name, p in paths.items() if name != 'result']})
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished_unix'] = time.time()
        write_new(paths['result'], result)
    return result


def comparison_inputs(argv, flag, following):
    return argv[argv.index(flag) + 1:argv.index(following)]


def execute_comparison(plan, comparison, executor_sha256):
    argv = comparison['argv']
    local = comparison_inputs(argv, '--local', '--stock')
    stock = comparison_inputs(argv, '--stock', '--workloads')
    jobs = [job for job in plan['jobs'] if job['mode'] == comparison['mode']]
    if (set(local) != {job['output'] for job in jobs if job['engine'] == 'local'}
            or set(stock) != {job['output'] for job in jobs if job['engine'] == 'stock'}
            or len(local + stock) != len(set(local + stock))):
        raise ValueError('comparison omits or duplicates a planned suite')
    warmup = int(option(argv, '--warmup'))
    samples = plan['measured_samples_per_candidate']
    if any(job['warmups_per_workload'] != warmup for job in jobs) or int(option(argv, '--min-samples')) != samples:
        raise ValueError('comparison changes the planned warmup/sample policy')
    inputs = {name: sha(host_path(plan, name)) for name in local + stock}
    signature = fingerprint(plan, dict(comparison=comparison, measurements_sha256=inputs))
    output = host_path(plan, option(argv, '--output'))
    receipt = output.with_suffix('.result.json')
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (saved.get('complete') is not True or saved['job_sha256'] != signature
                or saved['executor_sha256'] != executor_sha256 or saved['output_sha256'] != sha(output)):
            raise ValueError('prior comparison is failed or changed')
        return saved
    if output.exists():
        raise RuntimeError('unverified prior comparison; preserve it and use a fresh output path')
    result = dict(complete=False, job_sha256=signature, executor_sha256=executor_sha256)
    try:
        with output.with_suffix('.stdout.log').open('x') as out, output.with_suffix('.stderr.log').open('x') as err:
            result['exit_code'] = subprocess.run(argv, cwd=plan['cwd'], stdout=out, stderr=err).returncode
        if result['exit_code'] != 0:
            raise RuntimeError('comparison process failed')
        report = json.loads(output.read_text())
        if (report['input_sha256'] != inputs or report['workloads'] != plan['scenarios']
                or len(report['stock_candidates']) != comparison['expected_stock_candidates']
                or len(report['local_candidates']) != 1
                or report['minimum_measured_samples'] != samples or report['warmups_per_file'] != warmup
                or any(values['n'] != samples for candidate in report['local_candidates'] + report['stock_candidates']
                       for values in candidate['workloads'].values())):
            raise ValueError('comparison coverage differs from plan')
        result.update(complete=True, output_sha256=sha(output), all_local_medians_lower=report['all_local_medians_lower'])
    finally:
        write_new(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Run after candidate validation; default is input/image preflight only')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['format_version'] != 1 or not plan['jobs'] or len({j['output'] for j in plan['jobs']}) != len(plan['jobs']):
        raise ValueError('invalid or duplicate plan jobs')
    # A held advisory lock prevents a second executor from acting on stale
    # progress. External GPU work is checked independently before every suite.
    with args.plan.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_inputs(plan)
        for job in plan['jobs']:
            validate_job(plan, job)
        if not args.execute:
            print(json.dumps(dict(input_preflight_passed=True, suites=len(plan['jobs']),
                                  scenarios=plan['scenarios'], gpu_execution_started=False)))
            return
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip().splitlines()
        if len(devices) != 1:
            raise ValueError('this plan requires one visible workstation GPU')
        executor_sha256 = sha(Path(__file__))
        for job in plan['jobs']:
            verify_inputs(plan)
            print('Suite', job['name'], flush=True)
            execute_suite(plan, job, devices[0], executor_sha256)
        for comparison in plan['comparisons']:
            verify_inputs(plan)
            execute_comparison(plan, comparison, executor_sha256)
        print('All planned suites and comparisons completed. Inspect rankings and finish the independent acceptance audit.', flush=True)


if __name__ == '__main__':
    main()
