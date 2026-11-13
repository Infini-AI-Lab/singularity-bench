"""Regressions for protected NVIDIA/host processes outside the target cgroup."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('breakwater_collector_scope', HERE / 'collector/process_scope.py')
scope = importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)


class CollectorScopeTests(unittest.TestCase):
    def test_descendants_match_but_neighbor_containers_and_prefix_collisions_do_not(self):
        boundary = [{'hierarchy':'0','controllers':'','path':'/docker/'+'a'*64}]
        for path, expected in ((boundary[0]['path'], True), (boundary[0]['path']+'/nested', True),
                               (boundary[0]['path']+'0',False),('/docker/'+'b'*64,False)):
            with patch.object(scope,'cgroups',return_value=[{'hierarchy':'0','controllers':'','path':path}]):
                self.assertEqual(scope.belongs(1,boundary),expected)

    def test_pinning_rejects_shared_root_and_requires_a_specific_container(self):
        with patch.object(scope,'cgroups',return_value=[{'path':'/'}]):
            with self.assertRaises(ValueError):scope.pin(1)
        row={'path':'/system.slice/docker-'+'a'*64+'.scope'}
        with patch.object(scope,'cgroups',return_value=[row,{'path':'/'}]):
            self.assertEqual(scope.pin(1),[row])

    def test_unrelated_denied_namespace_does_not_hide_target_cgroup_members(self):
        sys.path.insert(0,str(HERE))
        import process_guard
        from types import SimpleNamespace
        paths=[Path('/fake/proc/11/cgroup'),Path('/fake/proc/22/cgroup')]
        original={'hierarchy':'0','controllers':'','path':'/docker/task'}
        def groups(pid):return [original if pid==11 else {**original,'path':'/protected-host'}]
        with patch.object(process_guard,'Path',return_value=SimpleNamespace(glob=lambda _:paths)), \
             patch.object(process_guard,'cgroups',side_effect=groups), \
             patch.object(process_guard.os,'readlink',side_effect=PermissionError('protected host')):
            self.assertEqual(process_guard.remaining({'cgroups':[original],'pid_namespace':'pid:[task]'}),[11])


if __name__=='__main__':unittest.main()
