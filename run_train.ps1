$env:PATH = "C:\Users\kavip\Python311Embed\Lib\site-packages\torch\lib;" + $env:PATH
$env:PYTHONPATH = "C:\Users\kavip\Downloads\hack"
Set-Location "C:\Users\kavip\Downloads\hack"
& "C:\Users\kavip\Python311Embed\python.exe" -W ignore -c "
import sys
sys.path.insert(0, r'C:\Users\kavip\Downloads\hack')
import runpy
sys.argv = sys.argv[1:]
runpy.run_path(r'C:\Users\kavip\Downloads\hack\train.py', run_name='__main__')
" @args
