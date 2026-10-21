"""Exercise failure/recovery boundaries without GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import execute_batch_plan as runner
import test_compare_batch_performance as fixtures


class ExecutionPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sample = fixtures.PerformanceComparisonTests()
        sample.setUp()
        self.records, self.stock, self.workloads = sample.local, sample.slow, sample.workloads
        (self.root / 'model').mkdir()
        for name, value in [('workloads.json', self.workloads), ('config.json', self.records[0]['engine_config']),
                            ('model/manifest.json', {'test': 'independent fixture identity'})]:
            (self.root / name).write_text(json.dumps(value))
        (self.root / 'binary').write_bytes(b'not an executable')
        self.plan = dict(cwd=str(self.root), inputs_sha256={name: runner.sha(self.root / name) for name in
                         ['workloads.json', 'config.json', 'model/manifest.json', 'binary']},
                         images={}, environment={}, scenarios=2, measured_samples_per_candidate=2)
        for record in self.records + self.stock:
            record['model_root'] = self.plan['inputs_sha256']['model/manifest.json']
        for record in self.records:
            record['binary_sha256'] = self.plan['inputs_sha256']['binary']
        self.job = dict(name='local-a1', mode='cold-reuse', engine='local', output='local.jsonl',
                        expected_records=6, warmups_per_workload=1, measured_per_workload=2,
                        argv=['fake', '--binary', 'binary', '--model', 'model', '--config', 'config.json',
                              '--workloads', 'workloads.json', '--output', 'local.jsonl', '--cache-mode', 'reuse', '--repeat', '3'])
        self.plan['jobs'] = [self.job]
        self.output = self.root / self.job['output']

    def save_records(self, path, records):
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))

    def fake_run(self, plan, job, paths):
        self.save_records(self.output, self.records)
        for name in ['stdout', 'stderr', 'telemetry_stderr']:
            paths[name].write_text('')
        paths['process'].write_text(json.dumps(dict(pid=123, argv=job['argv'])))
        paths['telemetry'].write_text('time,name,util,memory,sm,mem,temp,power\n0,test-gpu,1,2,3,4,5,6\n')
        return 0

    def execute(self, side_effect=None):
        with patch.object(runner, 'require_idle_gpu'), patch.object(runner, 'run_measured', side_effect=side_effect or self.fake_run):
            return runner.execute_suite(self.plan, self.job, 'test-gpu', 'executor-sha')

    def test_complete_suite_is_revalidated_and_resumed_without_execution(self):
        saved = self.execute()
        self.assertTrue(saved['complete'])
        with patch.object(runner, 'require_idle_gpu') as idle, patch.object(runner, 'run_measured') as run:
            self.assertEqual(runner.execute_suite(self.plan, self.job, 'test-gpu', 'executor-sha'), saved)
        idle.assert_not_called()
        run.assert_not_called()

    def test_complete_looking_output_with_nonzero_exit_is_never_accepted(self):
        def failure(plan, job, paths):
            self.fake_run(plan, job, paths)
            return 9
        with self.assertRaisesRegex(RuntimeError, 'exited 9'):
            self.execute(failure)
        self.assertFalse(json.loads(runner.paths_for(self.output)['result'].read_text())['complete'])
        with self.assertRaisesRegex(RuntimeError, 'failed or changed'):
            self.execute()

    def test_partial_or_active_artifacts_cannot_trigger_a_restart(self):
        self.save_records(self.output, self.records[:1])
        with patch.object(runner, 'require_idle_gpu') as idle, patch.object(runner, 'run_measured') as run:
            with self.assertRaisesRegex(RuntimeError, 'partial or active'):
                runner.execute_suite(self.plan, self.job, 'test-gpu', 'executor-sha')
        idle.assert_not_called()
        run.assert_not_called()

    def test_bad_command_binding_is_rejected_before_gpu_or_artifact_creation(self):
        self.job['argv'][self.job['argv'].index('--output') + 1] = 'wrong.jsonl'
        with patch.object(runner, 'require_idle_gpu') as idle, patch.object(runner, 'run_measured') as run:
            with self.assertRaisesRegex(ValueError, 'command output'):
                runner.execute_suite(self.plan, self.job, 'test-gpu', 'executor-sha')
        idle.assert_not_called()
        run.assert_not_called()
        self.assertFalse(runner.paths_for(self.output)['started'].exists())

    def test_changed_output_or_pinned_input_cannot_reuse_success(self):
        self.execute()
        with self.output.open('a') as out:
            out.write('\n')
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            self.execute()
        (self.root / 'binary').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'pinned input changed'):
            runner.verify_inputs(self.plan)

    def test_matching_internal_metadata_must_also_match_the_frozen_command(self):
        for record in self.records:
            record['binary_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'binary/configuration'):
            self.execute()

    def test_short_output_or_absent_telemetry_cannot_pass(self):
        def incomplete(plan, job, paths):
            self.fake_run(plan, job, paths)
            self.save_records(self.output, self.records[:-1])
            return 0
        with self.assertRaisesRegex(ValueError, 'record count'):
            self.execute(incomplete)
        # Fresh output path preserves the failed attempt and its receipt.
        self.output = self.root / 'retry.jsonl'
        self.job['output'] = 'retry.jsonl'
        self.job['argv'][self.job['argv'].index('--output') + 1] = 'retry.jsonl'
        def no_telemetry(plan, job, paths):
            self.fake_run(plan, job, paths)
            paths['telemetry'].write_text('')
            return 0
        with self.assertRaisesRegex(ValueError, 'telemetry'):
            self.execute(no_telemetry)

    def test_stock_identity_is_checked_against_the_planned_image_and_configuration(self):
        image = 'vllm/image@sha256:' + 'd' * 64
        self.plan['images'] = {'vllm': dict(reference=image, image_id='sha256:' + 'e' * 64)}
        job = dict(self.job, engine='stock', candidate=dict(image='vllm'), output='stock.jsonl',
                   argv=['fake', '--model', 'model/hf', '--requests', 'workloads.json', '--output', 'stock.jsonl',
                         '--cache-mode', 'reuse', '--repeat', '3', '--linear-backend', 'triton',
                         '--attention-backend', 'TRITON_ATTN', '--quantization', 'compressed-tensors',
                         '--hardware-profile', 'rtx-pro-6000', '--max-context', '32768',
                         '--max-num-seqs', '32', '--kv-cache-bytes', str(12 << 30)])
        for record in self.stock:
            record.update(linear_backend='triton', eager=False, graph_size_one=False, graph_capture_sizes=None,
                          graphs_without_compile=False, disable_cutedsl_warmup=False, memory_fraction=0.30,
                          environment={'INFERROUS_STOCK_IMAGE': image})
        self.save_records(self.root / 'stock.jsonl', self.stock)
        runner.validate_suite(self.plan, job, 'test-gpu')
        for record in self.stock:
            record['environment']['INFERROUS_STOCK_IMAGE'] = 'different-image'
        self.save_records(self.root / 'stock.jsonl', self.stock)
        with self.assertRaisesRegex(ValueError, 'stock configuration/image'):
            runner.validate_suite(self.plan, job, 'test-gpu')

    def test_comparison_uses_complete_planned_arms_and_can_resume_its_receipt(self):
        self.save_records(self.output, self.records)
        self.save_records(self.root / 'stock.jsonl', self.stock)
        self.plan['jobs'].append(dict(name='stock-b1', mode='cold-reuse', engine='stock', output='stock.jsonl', warmups_per_workload=1))
        comparison = dict(mode='cold-reuse', expected_stock_candidates=1,
                          argv=[sys.executable, str(Path(runner.__file__).with_name('compare_batch_performance.py')),
                                '--local', 'local.jsonl', '--stock', 'stock.jsonl', '--workloads', 'workloads.json',
                                '--warmup', '1', '--min-samples', '2', '--output', 'comparison.json'])
        saved = runner.execute_comparison(self.plan, comparison, 'executor-sha')
        self.assertTrue(saved['complete'])
        self.assertEqual(runner.execute_comparison(self.plan, comparison, 'executor-sha'), saved)
        omitted = copy.deepcopy(comparison)
        omitted['argv'].remove('stock.jsonl')
        with self.assertRaisesRegex(ValueError, 'omits or duplicates'):
            runner.execute_comparison(self.plan, omitted, 'executor-sha')
        changed_warmup = copy.deepcopy(comparison)
        changed_warmup['argv'][changed_warmup['argv'].index('--warmup') + 1] = '0'
        with self.assertRaisesRegex(ValueError, 'warmup/sample policy'):
            runner.execute_comparison(self.plan, changed_warmup, 'executor-sha')


if __name__ == '__main__':
    unittest.main()
