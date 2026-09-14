import py_compile
from pathlib import Path
import tempfile
import unittest


class SourceCompileTest(unittest.TestCase):
    def test_agent_source_compiles(self):
        source = Path(__file__).resolve().parents[1] / 'multiagent/agent.py'
        with tempfile.TemporaryDirectory() as directory:
            py_compile.compile(source, cfile=str(Path(directory) / 'agent.pyc'), doraise=True)


if __name__ == '__main__':
    unittest.main()
