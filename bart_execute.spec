# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['bart_execute.py'],
             pathex=['/Users/houliu/Documents/Projects/inscriptio'],
             binaries=[],
             datas=[
                 ("./training", "./training"), 
                 ("./distenv/lib/python3.9/site-packages", ".")
             ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='bart_execute',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='bart_execute')
