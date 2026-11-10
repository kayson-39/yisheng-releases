import os
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runtime_app = Path(sys.executable).resolve().parents[2]
# Prefer the runtime actually executing this script. Sandboxed shells may map
# LOCALAPPDATA to another profile even when using the installed Python binary.
installed = (runtime_app if (runtime_app / 'runtime/packages').is_dir()
             else Path(os.environ['LOCALAPPDATA']) / 'Programs/YishengTranslatorWhite/app')
if not (installed / 'runtime/packages').is_dir():
    raise SystemExit('找不到译声运行依赖，请使用安装目录 app/runtime/py312/python.exe 运行。')
sys.path[:0] = [str(root), str(installed / 'runtime/packages')]
os.environ['JPSUB_DATA_DIR'] = str(root / 'artifacts/test-userdata')
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.dont_write_bytecode = True
suite = unittest.defaultTestLoader.discover(str(root/'tests'), pattern=sys.argv[1] if len(sys.argv)>1 else 'test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
