import os, sys

_plugin_dir = os.path.dirname(os.path.abspath(__file__))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

# Ajouter gs_c4d site-packages pour que requests soit disponible dans C4D
_gs_site = r"C:\Users\MSI\scoop\apps\miniconda3\current\envs\gs_c4d\Lib\site-packages"
if os.path.isdir(_gs_site) and _gs_site not in sys.path:
    sys.path.insert(0, _gs_site)

import c4d
from c4d import plugins, gui
from ui.main_dialog import GaussianEditorDialog

PLUGIN_ID = 1000001
_dialog = None

class GaussianEditorCommand(plugins.CommandData):
    def Execute(self, doc):
        global _dialog
        if _dialog is None:
            _dialog = GaussianEditorDialog()
        if not _dialog.IsOpen():
            _dialog.Open(dlgtype=c4d.DLG_TYPE_ASYNC, pluginid=PLUGIN_ID,
                        defaultw=420, defaulth=650, xpos=-2, ypos=-2)
        else:
            _dialog.Close()
        return True

    def RestoreLayout(self, sec_ref):
        global _dialog
        if _dialog is None:
            _dialog = GaussianEditorDialog()
        return _dialog.Restore(PLUGIN_ID, sec_ref)

    def GetState(self, doc):
        state = c4d.CMD_ENABLED
        if _dialog is not None and _dialog.IsOpen():
            state |= c4d.CMD_VALUE
        return state

if __name__ == '__main__':
    ok = plugins.RegisterCommandPlugin(
        id=PLUGIN_ID,
        str='GaussianEditor',
        info=c4d.PLUGINFLAG_COMMAND_HOTKEY,
        icon=None,
        help='Gaussian Splatting Editor',
        dat=GaussianEditorCommand()
    )
    print('[GaussianEditor] Charge OK' if ok else '[GaussianEditor] ERREUR')