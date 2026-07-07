"""
ui/main_dialog.py
------------------
Interface principale GaussianEditor C4D — version simplifiée.
Compatible C4D 2025 (CPython 3.11).

Fix juillet 2026 v3 :
  - _check_scene_loaded() vérifie serveur ET C4D
  - _sync_to_server() synchronise le PLY vers le serveur après import
  - Import PLY → chargement automatique dans le serveur si backend actif
"""

import os
import sys
import threading
import time

try:
    import c4d
    from c4d import gui
    INSIDE_C4D = True
except ImportError:
    INSIDE_C4D = False

_plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

from core.ply_io         import parse_gaussian_ply, write_gaussian_ply
from core.gsplat_viewer  import (gaussians_to_c4d, c4d_to_gaussians,
                                  get_selected_gaussian_obj)
from core.backend_gsplat import GsplatBackend

PLUGIN_ID = 1000001

EVT_IMPORT_DONE   = 1001
EVT_ACTION_DONE   = 1002
EVT_ACTION_ERROR  = 1003
EVT_LOG_MSG       = 1004
EVT_BACKEND_READY = 1005


class UI:
    BTN_IMPORT        = 2101
    BTN_EXPORT        = 2102
    EDT_IMAGE_PATH    = 2151
    BTN_IMAGE_BROWSE  = 2152
    BTN_GENERATE      = 2156
    EDT_PROMPT        = 2201
    BTN_TRACE         = 2203
    BTN_EDIT          = 2204
    BTN_DELETE        = 2205
    BTN_CROP          = 2206
    BTN_UNDO          = 2207
    BTN_SAVE          = 2209
    EDT_SAVE_PATH     = 2210
    BTN_EXPORT_SPLAT  = 2215
    EDT_PYTHON_PATH   = 2303
    BTN_PYTHON_BROWSE = 2304
    BTN_BACKEND_START = 2305
    BTN_BACKEND_STOP  = 2306
    TXT_BACKEND_STATE = 2307
    EDT_LOG           = 2400
    BTN_CLEAR_LOG     = 2401
    TXT_STATUS        = 2500
    TXT_INFO_N        = 2501
    TXT_INFO_FILE     = 2502


class GaussianEditorDialog(gui.GeDialog):

    def __init__(self):
        super().__init__()
        self._backend            = None
        self._current_gd         = None
        self._current_result_ply = ""
        self._current_ply_path   = ""
        self._is_importing       = False
        self._log_lines          = []
        self._pending_log        = ""

    def CreateLayout(self):
        self.SetTitle("GaussianEditor for C4D")

        # Fichier
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1,
                           title="Fichier", groupflags=c4d.BORDER_GROUP_IN):
            self.GroupBorderSpace(8, 6, 8, 6)
            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2):
                self.AddButton(UI.BTN_IMPORT, c4d.BFH_SCALEFIT,
                               inith=24, name="Importer .ply")
                self.AddButton(UI.BTN_EXPORT, c4d.BFH_SCALEFIT,
                               inith=24, name="Exporter .ply")
            self.GroupEnd()
        self.GroupEnd()

        # Generation
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1,
                           title="Generer depuis image (TripoSplat)",
                           groupflags=c4d.BORDER_GROUP_IN):
            self.GroupBorderSpace(8, 6, 8, 6)
            self.AddStaticText(0, c4d.BFH_LEFT, name="Image source :")
            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2):
                self.AddEditText(UI.EDT_IMAGE_PATH, c4d.BFH_SCALEFIT)
                self.AddButton(UI.BTN_IMAGE_BROWSE, c4d.BFH_RIGHT,
                               initw=36, name="...")
            self.GroupEnd()
            self.AddButton(UI.BTN_GENERATE, c4d.BFH_SCALEFIT,
                           inith=30, name="Generer")
        self.GroupEnd()

        # Actions IA
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1,
                           title="Actions IA", groupflags=c4d.BORDER_GROUP_IN):
            self.GroupBorderSpace(8, 6, 8, 6)
            self.AddStaticText(0, c4d.BFH_LEFT,
                               name="Prompt (segmentation / edition) :")
            self.AddEditText(UI.EDT_PROMPT, c4d.BFH_SCALEFIT)

            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2):
                self.AddButton(UI.BTN_TRACE, c4d.BFH_SCALEFIT,
                               inith=24, name="Segmenter")
                self.AddButton(UI.BTN_EDIT,  c4d.BFH_SCALEFIT,
                               inith=24, name="Editer")
            self.GroupEnd()
            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=3):
                self.AddButton(UI.BTN_DELETE, c4d.BFH_SCALEFIT,
                               inith=24, name="Supprimer")
                self.AddButton(UI.BTN_CROP,   c4d.BFH_SCALEFIT,
                               inith=24, name="Crop")
                self.AddButton(UI.BTN_UNDO,   c4d.BFH_SCALEFIT,
                               inith=24, name="Undo")
            self.GroupEnd()
            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=3):
                self.AddEditText(UI.EDT_SAVE_PATH, c4d.BFH_SCALEFIT)
                self.AddButton(UI.BTN_SAVE,
                               c4d.BFH_RIGHT, initw=60, name="Sauver")
                # self.AddButton(UI.BTN_EXPORT_SPLAT,
                #                c4d.BFH_RIGHT, initw=60, name=".splat")
            self.GroupEnd()
        self.GroupEnd()

        # Backend
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1,
                           title="Backend", groupflags=c4d.BORDER_GROUP_IN):
            self.GroupBorderSpace(8, 6, 8, 6)
            self.AddStaticText(0, c4d.BFH_LEFT,
                               name="Python externe (gs_c4d) :")
            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=2):
                self.AddEditText(UI.EDT_PYTHON_PATH, c4d.BFH_SCALEFIT)
                self.AddButton(UI.BTN_PYTHON_BROWSE, c4d.BFH_RIGHT,
                               initw=36, name="...")
            self.GroupEnd()
            self.AddSeparatorH(0)
            if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=3):
                self.AddButton(UI.BTN_BACKEND_START, c4d.BFH_LEFT,
                               initw=100, inith=24, name="Demarrer")
                self.AddButton(UI.BTN_BACKEND_STOP,  c4d.BFH_LEFT,
                               initw=100, inith=24, name="Arreter")
                self.AddStaticText(UI.TXT_BACKEND_STATE,
                                   c4d.BFH_SCALEFIT, name="Arrete")
            self.GroupEnd()
        self.GroupEnd()

        # Info scène
        self.AddSeparatorH(0)
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=4):
            self.GroupBorderSpace(8, 2, 8, 2)
            self.AddStaticText(0, c4d.BFH_LEFT, name="Scene :")
            self.AddStaticText(UI.TXT_INFO_FILE, c4d.BFH_SCALEFIT, name="--")
            self.AddStaticText(0, c4d.BFH_LEFT, name="Points :")
            self.AddStaticText(UI.TXT_INFO_N, c4d.BFH_LEFT, name="--")
        self.GroupEnd()

        # Log
        self.AddSeparatorH(0)
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1,
                           title="Log", groupflags=c4d.BORDER_GROUP_IN):
            self.GroupBorderSpace(6, 4, 6, 4)
            self.AddMultiLineEditText(
                UI.EDT_LOG,
                c4d.BFH_SCALEFIT | c4d.BFV_SCALEFIT,
                initw=0, inith=100,
                style=c4d.DR_MULTILINE_READONLY | c4d.DR_MULTILINE_MONOSPACED
            )
            self.AddButton(UI.BTN_CLEAR_LOG, c4d.BFH_RIGHT,
                           initw=60, inith=16, name="Vider")
        self.GroupEnd()

        # Statut
        self.AddSeparatorH(0)
        if self.GroupBegin(0, c4d.BFH_SCALEFIT, cols=1):
            self.GroupBorderSpace(8, 2, 8, 2)
            self.AddStaticText(UI.TXT_STATUS, c4d.BFH_SCALEFIT, name="Pret.")
        self.GroupEnd()

        return True

    def InitValues(self):
        self.SetString(UI.EDT_SAVE_PATH, "ui_result/result.ply")

        # 1. Charger depuis les preferences C4D
        python_path = ""
        bc = c4d.plugins.GetWorldPluginData(PLUGIN_ID)
        if bc:
            python_path = bc.GetString(2) or ""

        # 2. Fallback : lire python_path.txt genere par install.sh
        if not python_path:
            plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            txt_path   = os.path.join(plugin_dir, "python_path.txt")
            if os.path.isfile(txt_path):
                try:
                    with open(txt_path, "r") as fp:
                        python_path = fp.read().strip()
                except Exception:
                    pass

        if python_path:
            self.SetString(UI.EDT_PYTHON_PATH, python_path)

        return True

    def Command(self, id, msg):
        if   id == UI.BTN_IMPORT:        self._do_import()
        elif id == UI.BTN_EXPORT:        self._do_export()
        elif id == UI.BTN_IMAGE_BROWSE:  self._browse_image()
        elif id == UI.BTN_GENERATE:      self._do_generate()
        elif id == UI.BTN_TRACE:         self._require_backend(self._do_trace)
        elif id == UI.BTN_EDIT:          self._require_backend(self._do_edit)
        elif id == UI.BTN_DELETE:        self._require_backend(self._do_delete)
        elif id == UI.BTN_CROP:          self._require_backend(self._do_crop)
        elif id == UI.BTN_UNDO:          self._require_backend(self._do_undo)
        elif id == UI.BTN_SAVE:          self._require_backend(self._do_save)
        elif id == UI.BTN_EXPORT_SPLAT:  self._require_backend(self._do_export_splat)
        elif id == UI.BTN_PYTHON_BROWSE: self._browse_python()
        elif id == UI.BTN_BACKEND_START: self._do_backend_start()
        elif id == UI.BTN_BACKEND_STOP:  self._do_backend_stop()
        elif id == UI.BTN_CLEAR_LOG:
            self._log_lines.clear()
            self.SetString(UI.EDT_LOG, "")
        return True

    def CoreMessage(self, id, msg):
        if id == EVT_IMPORT_DONE:
            separate = bool(msg.GetInt32(c4d.BFM_CORE_PAR1))
            self._finalize_import(separate=separate)
        elif id == EVT_ACTION_DONE:
            self._set_status("Pret.")
            c4d.gui.StatusSetText("")
        elif id == EVT_ACTION_ERROR:
            self._set_status("Erreur — voir le log")
            c4d.gui.StatusSetText("")
        elif id == EVT_LOG_MSG:
            if self._pending_log:
                self._append_log(self._pending_log)
                self._pending_log = ""
        elif id == EVT_BACKEND_READY:
            self.SetString(UI.TXT_BACKEND_STATE, "Pret")
            self._set_status("Backend pret.")
            c4d.gui.StatusSetText("")
            self._auto_load_scene()
        return super().CoreMessage(id, msg)

    # ─────────────────────────────────────────────────────────
    # Import / Export
    # ─────────────────────────────────────────────────────────

    def _do_import(self):
        if self._is_importing:
            return
        path = c4d.storage.LoadDialog(
            title="Importer Gaussian Splatting (.ply)",
            flags=c4d.FILESELECT_LOAD, force_suffix="ply"
        )
        if not path:
            return
        self._is_importing     = True
        self._current_ply_path = path
        self._set_status(f"Chargement {os.path.basename(path)}...")
        self._append_log(f"[Import] {os.path.basename(path)}")

        def worker():
            try:
                gd = parse_gaussian_ply(path)
                self._current_gd = gd
                c4d.SpecialEventAdd(EVT_IMPORT_DONE)
            except Exception as e:
                self._append_log(f"[Import] Erreur : {e}")
                self._is_importing = False

        threading.Thread(target=worker, daemon=True).start()

    def _finalize_import(self, separate=False):
        try:
            gd   = self._current_gd
            doc  = c4d.documents.GetActiveDocument()
            src  = gd.source_path or self._current_ply_path or ""
            name = os.path.splitext(os.path.basename(src))[0] if src else "gaussians"

            doc.StartUndo()

            # Mode séparé : garder les objets existants, juste ajouter le nouveau
            if not separate:
                for rname in ["edited", "deleted", "cropped", "added",
                              "opacity", "scaled", "result", "undo", name]:
                    old_obj = doc.SearchObject(rname)
                    if old_obj:
                        doc.AddUndo(c4d.UNDOTYPE_DELETEOBJ, old_obj)
                        old_obj.Remove()

            obj = gaussians_to_c4d(doc, gd, name=name)
            doc.InsertObject(obj)
            doc.AddUndo(c4d.UNDOTYPE_NEWOBJ, obj)
            doc.EndUndo()
            doc.SetActiveObject(obj)
            c4d.EventAdd()

            self.SetString(UI.TXT_INFO_FILE, os.path.basename(src))
            self.SetString(UI.TXT_INFO_N,    f"{gd.n:,}")
            mode_label = " (ajout)" if separate else ""
            self._set_status(f"Importe{mode_label} : {gd.n:,} points")

            # Synchroniser vers le serveur si backend actif
            if self._backend and self._backend.is_ready and self._current_ply_path:
                self._sync_to_server(self._current_ply_path)

        except Exception as e:
            self._set_status(f"Erreur C4D : {e}")
            self._append_log(f"[Import] Erreur finalize : {e}")
        finally:
            self._is_importing = False

    def _do_export(self):
        doc = c4d.documents.GetActiveDocument()
        obj = get_selected_gaussian_obj(doc) or doc.GetActiveObject()
        if not isinstance(obj, c4d.PointObject):
            gui.MessageDialog("Selectionnez un objet GS.", c4d.GEMB_OK)
            return
        path = c4d.storage.SaveDialog(
            title="Exporter .ply", flags=c4d.FILESELECT_SAVE,
            force_suffix="ply", def_file=obj.GetName() + ".ply"
        )
        if not path:
            return
        def worker():
            try:
                gd = c4d_to_gaussians(obj)
                write_gaussian_ply(path, gd)
                self._set_status(f"Exporte : {gd.n:,} pts")
            except Exception as e:
                self._set_status(f"Erreur export : {e}")
        threading.Thread(target=worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # Génération — logique automatique nouvelle/ajouter
    # ─────────────────────────────────────────────────────────

    def _sync_to_server(self, ply_path: str):
        """
        Charge un PLY dans le serveur en arrière-plan.
        Appelé après import et après démarrage du backend.
        """
        def _load():
            try:
                import requests
                wsl    = self._win_to_wsl(ply_path)
                r      = requests.post("http://127.0.0.1:8086/load",
                                       json={"ply_path": wsl}, timeout=30)
                result = r.json()
                if result.get("ok"):
                    self._append_log(
                        f"[Serveur] Scene synchronisee : "
                        f"{result.get('n', 0):,} gaussiens"
                    )
                else:
                    self._append_log(
                        f"[Serveur] Sync echouee : {result.get('error')}"
                    )
            except Exception as e:
                self._append_log(f"[Serveur] Sync echouee : {e}")
        threading.Thread(target=_load, daemon=True).start()

    def _check_scene_loaded(self) -> bool:
        """
        Vérifie si une scène est chargée.

        Priorité :
          1. Interroger le serveur → n > 0
          2. Si serveur vide mais PLY courant disponible
             → le charger dans le serveur puis retourner True
          3. Sinon → False (nouvelle scène)
        """
        try:
            import requests
            r    = requests.get("http://127.0.0.1:8086/scene_info", timeout=3)
            data = r.json()
            if data.get("n", 0) > 0:
                return True
        except Exception:
            pass

        # Serveur vide mais PLY courant disponible → synchroniser
        if self._current_ply_path and os.path.isfile(self._current_ply_path):
            self._append_log(
                f"[Serveur] Scene absente — synchronisation "
                f"{os.path.basename(self._current_ply_path)}..."
            )
            self._sync_to_server(self._current_ply_path)
            # Attendre que la sync soit faite (max 10s)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    import requests
                    r    = requests.get("http://127.0.0.1:8086/scene_info", timeout=2)
                    data = r.json()
                    if data.get("n", 0) > 0:
                        self._append_log(
                            f"[Serveur] Scene prete : {data['n']:,} gaussiens"
                        )
                        return True
                except Exception:
                    pass

        return False

    def _do_generate(self):
        image_path = self.GetString(UI.EDT_IMAGE_PATH).strip()
        if not image_path or not os.path.isfile(image_path):
            gui.MessageDialog(
                "Image introuvable ou non selectionnee.", c4d.GEMB_OK
            )
            return
        if not self._backend or not self._backend.is_ready:
            gui.MessageDialog(
                "Backend non demarre.\n"
                "1. Entrez le chemin Python gs_c4d\n"
                "2. Cliquez Demarrer",
                c4d.GEMB_OK
            )
            return

        wsl_path = self._win_to_wsl(image_path)

        # Décision automatique : nouvelle scène ou ajouter
        scene_loaded = self._check_scene_loaded()

        if scene_loaded:
            mode        = "Ajout a la scene (auto)"
            endpoint_fn = lambda on_done, on_error: \
                self._backend.generate_and_add(
                    wsl_path, 0, on_done=on_done, on_error=on_error
                )
        else:
            mode        = "Nouvelle scene (auto)"
            endpoint_fn = lambda on_done, on_error: \
                self._backend.generate(
                    wsl_path, 0, on_done=on_done, on_error=on_error
                )

        self._set_status(f"Generation TripoSplat ({mode}, ~13-30 min)...")
        c4d.gui.StatusSetSpin()
        self._append_log(f"[Generation] Image : {os.path.basename(image_path)}")
        self._append_log(f"[Generation] Mode  : {mode}")

        def on_done(result):
            if result.get("ok"):
                ply = result.get("ply_path", "")
                n   = result.get("n", 0)
                self._append_log(
                    f"[Generation] PLY : {os.path.basename(ply)} ({n:,} gaussians)"
                )
                self._on_done(result, "Generation terminee")
            else:
                self._on_error(result.get("error", "Erreur inconnue"))

        endpoint_fn(on_done, self._on_error)

    # ─────────────────────────────────────────────────────────
    # Actions IA
    # ─────────────────────────────────────────────────────────

    def _require_backend(self, fn):
        if self._backend is None or not self._backend.is_ready:
            gui.MessageDialog(
                "Backend non demarre.\n"
                "1. Entrez le chemin Python\n"
                "2. Cliquez Demarrer",
                c4d.GEMB_OK
            )
            return
        fn()

    def _do_trace(self):
        prompt = self.GetString(UI.EDT_PROMPT).strip()
        if not prompt:
            gui.MessageDialog("Entrez un prompt.", c4d.GEMB_OK)
            return
        self._set_status(f"Segmentation '{prompt}'...")
        self._append_log(f"[Trace] '{prompt}'")
        c4d.gui.StatusSetSpin()
        self._backend.trace(
            prompt, 0.5,
            on_done=lambda r: self._on_done(r, "Segmentation terminee"),
            on_error=self._on_error
        )

    def _do_edit(self):
        prompt = self.GetString(UI.EDT_PROMPT).strip()
        if not prompt:
            gui.MessageDialog("Entrez un prompt.", c4d.GEMB_OK)
            return
        self._set_status(f"Edition '{prompt}'...")
        self._append_log(f"[Edit] '{prompt}'")
        c4d.gui.StatusSetSpin()
        self._backend.edit(
            prompt,
            on_done=lambda r: self._on_done(r, "Edition terminee"),
            on_error=self._on_error
        )

    def _do_delete(self):
        if not gui.QuestionDialog("Supprimer la selection ?"):
            return
        self._set_status("Suppression en cours...")
        self._append_log("[Delete] Suppression...")
        c4d.gui.StatusSetSpin()
        self._backend.delete(
            on_done=lambda r: self._on_done(r, "Suppression terminee"),
            on_error=self._on_error
        )

    def _do_crop(self):
        if not gui.QuestionDialog("Garder uniquement la selection ?"):
            return
        self._set_status("Crop en cours...")
        self._append_log("[Crop] Isolation de la selection...")
        c4d.gui.StatusSetSpin()
        self._backend.crop(
            on_done=lambda r: self._on_done(r, "Crop termine"),
            on_error=self._on_error
        )

    def _do_undo(self):
        self._set_status("Annulation...")
        self._append_log("[Undo] Restauration...")
        self._backend.undo(
            on_done=lambda r: self._on_done(r, "Undo effectue"),
            on_error=self._on_error
        )

    def _do_save(self):
        """
        Lit les transformations C4D de chaque objet GS et appelle
        /merge_and_save sur le serveur pour fusionner avec les vrais SH.
        """
        out = self.GetString(UI.EDT_SAVE_PATH).strip() or "result.ply"
        if not os.path.isabs(out) and not out.startswith("C:"):
            # Construire chemin absolu Windows
            base_dir = "C:\\Users\\MSI\\Documents\\C4D\\ui_result"
            if self._current_ply_path:
                base_dir = os.path.dirname(self._current_ply_path)
            out = os.path.join(base_dir, os.path.basename(out))

        if not self._backend or not self._backend.is_ready:
            gui.MessageDialog("Backend non demarre.", c4d.GEMB_OK)
            return

        doc = c4d.documents.GetActiveDocument()

        # Collecter tous les objets GS
        gs_objects = []
        obj = doc.GetFirstObject()
        while obj:
            try:
                bc = obj.GetDataInstance()
                if bc and bc.GetInt32(10002) > 0:
                    gs_objects.append(obj)
            except Exception:
                pass
            obj = obj.GetNext()

        if not gs_objects:
            gui.MessageDialog("Aucun objet GS dans la scene.", c4d.GEMB_OK)
            return

        self._append_log(f"[Save] {len(gs_objects)} objet(s) detectes")
        self._set_status(f"Preparation fusion {len(gs_objects)} objet(s)...")

        # Construire la liste des objets avec transformations C4D
        objects_list = []
        for gs_obj in gs_objects:
            try:
                bc       = gs_obj.GetDataInstance()
                src_path = bc.GetString(10001)  # GS_ATTR_SOURCE_PATH

                # Lire la matrice globale C4D
                mg = gs_obj.GetMg()

                # Lire position monde (après déplacement utilisateur)
                dx = float(mg.off.x)
                dy = float(mg.off.y)
                dz = float(mg.off.z)

                # Calculer le scale de normalisation utilisé dans gaussians_to_c4d
                # (200.0 / scene_size)
                n_pts = gs_obj.GetPointCount()
                if n_pts > 1:
                    pts = [gs_obj.GetPoint(i) for i in range(min(n_pts, 500))]
                    xs  = [p.x for p in pts]
                    ys  = [p.y for p in pts]
                    zs  = [p.z for p in pts]
                    c4d_size = max(
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                        max(zs) - min(zs)
                    )
                    # gaussians_to_c4d normalise à 200 unités
                    c4d_scale = 200.0 / c4d_size if c4d_size > 0.001 else 1.0
                else:
                    c4d_scale = 1.0

                # Scale utilisateur (mg.v1.GetLength() = 1.0 si pas de scale)
                user_scale = float(mg.v1.GetLength())

                objects_list.append({
                    "ply_path":  src_path,
                    "dx":        dx,
                    "dy":        dy,
                    "dz":        dz,
                    "scale":     user_scale,
                    "c4d_scale": c4d_scale,
                })
                self._append_log(
                    f"[Save] {gs_obj.GetName()} | "
                    f"pos=({dx:.1f},{dy:.1f},{dz:.1f}) | "
                    f"scale={user_scale:.3f}"
                )
            except Exception as e:
                self._append_log(f"[Save] Erreur lecture {gs_obj.GetName()} : {e}")

        if not objects_list:
            gui.MessageDialog("Impossible de lire les transformations.", c4d.GEMB_OK)
            return

        # Convertir chemin Windows → WSL pour out_path
        out_wsl = self._win_to_wsl(out)

        self._set_status("Fusion en cours via serveur...")
        c4d.gui.StatusSetSpin()

        self._backend._post_async(
            "/merge_and_save",
            {"objects": objects_list, "out_path": out_wsl},
            on_done  = lambda r: self._on_done(r, "Sauvegarde OK"),
            on_error = self._on_error,
            timeout  = 120,
        )


    def _do_export_splat(self):
        out = (self.GetString(UI.EDT_SAVE_PATH).strip()
               .replace(".ply", ".splat") or "ui_result/export.splat")
        self._set_status("Export .splat...")
        self._append_log(f"[Export] → {out}")
        self._backend.export_splat(
            out,
            on_done=lambda r: self._on_done(r, "Export .splat termine"),
            on_error=self._on_error
        )

    def _on_done(self, result, label):
        self._current_result_ply = result.get("ply_path", "")
        n_aff = (result.get("n_affected") or result.get("n_removed") or
                 result.get("n_added")    or result.get("n_selected") or
                 result.get("n_kept", 0))
        detail = f" ({n_aff:,} gaussiens)" if n_aff else ""
        self._append_log(f"[OK] {label}{detail}")
        self._set_status(f"OK : {label}{detail}")
        try:
            c4d.gui.StatusSetText("")
        except Exception:
            pass

        ply = self._wsl_to_win(self._current_result_ply)
        if not ply or not os.path.isfile(ply):
            ply = self._current_result_ply

        if ply and os.path.isfile(ply):
            age = time.time() - os.path.getmtime(ply)
            if age < 600:
                self._append_log(f"[Reimport] {os.path.basename(ply)}...")
                self._set_status(f"Reimport : {os.path.basename(ply)}...")

                def _reimport_with_flag(is_sep):
                    try:
                        self._append_log("[Reimport] Lecture PLY...")
                        gd = parse_gaussian_ply(ply)
                        self._append_log(
                            f"[Reimport] {gd.n:,} gaussiens -> C4D..."
                        )
                        self._current_gd       = gd
                        self._current_ply_path = ply
                        # p1=1 → objet séparé, p1=0 → remplace
                        c4d.SpecialEventAdd(EVT_IMPORT_DONE, p1=is_sep, p2=0)
                    except Exception as e:
                        self._append_log(f"[Reimport] Erreur : {e}")
                        c4d.SpecialEventAdd(EVT_ACTION_ERROR)

                is_separate = 1 if result.get("separate") else 0
                threading.Thread(
                    target=lambda sep=is_separate: _reimport_with_flag(sep),
                    daemon=True
                ).start()
            else:
                c4d.SpecialEventAdd(EVT_ACTION_DONE)
        else:
            c4d.SpecialEventAdd(EVT_ACTION_DONE)

    def _on_error(self, error):
        self._append_log(f"[ERREUR] {error}")
        c4d.SpecialEventAdd(EVT_ACTION_ERROR)

    # ─────────────────────────────────────────────────────────
    # Backend
    # ─────────────────────────────────────────────────────────

    def _do_backend_start(self):
        python = self.GetString(UI.EDT_PYTHON_PATH).strip()
        if not python or not os.path.isfile(python):
            gui.MessageDialog(
                "Entrez un chemin Python valide.\nExemple :\n"
                "C:\\Users\\MSI\\scoop\\apps\\miniconda3\\current\\"
                "envs\\gs_c4d\\python.exe",
                c4d.GEMB_OK
            )
            return
        self._save_prefs()
        self._backend = GsplatBackend(python_executable=python)
        self._backend.set_log_callback(self._on_backend_log)
        self.SetString(UI.TXT_BACKEND_STATE, "Demarrage...")
        self._set_status("Demarrage backend...")
        self._append_log("[Backend] Lancement via wsl.exe...")
        c4d.gui.StatusSetSpin()
        self._backend.start(
            ply_path = self._win_to_wsl(self._current_ply_path),
            on_ready = lambda: c4d.SpecialEventAdd(EVT_BACKEND_READY),
            on_error = lambda e: self._on_backend_error(e)
        )

    def _do_backend_stop(self):
        if self._backend:
            self._backend.stop()
            self._backend = None
        self.SetString(UI.TXT_BACKEND_STATE, "Arrete")
        self._set_status("Backend arrete.")
        c4d.gui.StatusSetText("")

    def _on_backend_log(self, msg):
        self._pending_log = msg
        c4d.SpecialEventAdd(EVT_LOG_MSG)

    def _on_backend_error(self, error):
        self._append_log(f"[ERREUR backend] {error}")
        self.SetString(UI.TXT_BACKEND_STATE, "Erreur")
        self._set_status("Erreur backend — voir log")
        c4d.gui.StatusSetText("")

    def _auto_load_scene(self):
        """Charge automatiquement la scène dans le serveur après Démarrer."""
        if not self._current_ply_path or not os.path.isfile(self._current_ply_path):
            return
        self._sync_to_server(self._current_ply_path)

    # ─────────────────────────────────────────────────────────
    # Browse
    # ─────────────────────────────────────────────────────────

    def _browse_image(self):
        p = c4d.storage.LoadDialog(
            title="Image source (JPG, PNG, WEBP)",
            flags=c4d.FILESELECT_LOAD
        )
        if p:
            self.SetString(UI.EDT_IMAGE_PATH, p)
            self._append_log(f"[Image] {os.path.basename(p)}")

    def _browse_python(self):
        p = c4d.storage.LoadDialog(
            "Executable Python gs_c4d", c4d.FILESELECT_LOAD
        )
        if p:
            self.SetString(UI.EDT_PYTHON_PATH, p)
            self._save_prefs()

    # ─────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────

    def _set_status(self, text):
        self.SetString(UI.TXT_STATUS, text)
        c4d.gui.StatusSetText(text)

    def _append_log(self, text):
        self._log_lines.append(text)
        if len(self._log_lines) > 300:
            self._log_lines = self._log_lines[-300:]
        self.SetString(UI.EDT_LOG, "\n".join(self._log_lines))

    def _win_to_wsl(self, path):
        if not path or path.startswith("/"):
            return path
        path = path.replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            return f"/mnt/{path[0].lower()}/{path[2:].lstrip('/')}"
        return path

    def _wsl_to_win(self, path):
        if not path or not path.startswith("/mnt/"):
            return path
        parts = path[5:].split("/", 1)
        drive = parts[0].upper()
        rest  = parts[1].replace("/", "\\") if len(parts) > 1 else ""
        return f"{drive}:\\{rest}"

    def _save_prefs(self):
        bc = c4d.BaseContainer()
        bc.SetString(2, self.GetString(UI.EDT_PYTHON_PATH))
        c4d.plugins.SetWorldPluginData(PLUGIN_ID, bc)

    def DestroyWindow(self):
        if self._backend:
            self._backend.stop()
        super().DestroyWindow()