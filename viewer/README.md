# AVBD Three.js Viewer

This viewer renders AVBD snapshots in a browser using three.js. It can run with built-in sample scenes, or connect to the native C++ app over `ws://127.0.0.1:8765` when the viewer bridge is enabled.

## Run The Viewer

```powershell
cd C:\code\avbd-demo3d\viewer
npm install
npm run dev -- --port 5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Build The Native Bridge

From `C:\code\avbd-demo3d`, configure and build with:

```powershell
cmd.exe /c "call ""C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"" && ""C:\Program Files\CMake\bin\cmake.exe"" -S . -B build-viewer-bridge-nmake -G ""NMake Makefiles"" -DCMAKE_BUILD_TYPE=Release -DAVBD_ENABLE_VIEWER_BRIDGE=ON && ""C:\Program Files\CMake\bin\cmake.exe"" --build build-viewer-bridge-nmake --config Release"
```

Then run:

```powershell
C:\code\avbd-demo3d\build-viewer-bridge-nmake\avbd_demo3d.exe
```

## Smoke Checks

Validate bundled sample snapshots:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:samples
```

Validate browser-test hooks and diagnostics DOM ids:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:contracts
```

After `npm run build`, validate the built viewer in headless Edge or Chrome:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:browser
```

By default this writes a screenshot to `C:\tmp\avbd-three-viewer-smoke.png`.
The smoke test asks Windows for free localhost ports before using them.

With the C++ app running and the viewer bridge enabled, validate browser rendering of live native snapshots:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:browser:native
```

For native scene coverage, launch the C++ app with a startup scene override before running the smoke:

```powershell
$env:AVBD_START_SCENE = "Sphere Pour on Cylinders"
C:\code\avbd-demo3d\build-viewer-bridge-nmake\avbd_demo3d.exe
```

Then in another terminal:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:browser:native -- --expect-shapes sphere,cylinder
```

Validate the native bridge after launching the C++ app:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:bridge
```

The viewer retries the bridge connection automatically, so it is fine to open the browser before launching the native app.
