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

To run the simulation without the OpenGL debug window, launch the headless server instead:

```powershell
C:\code\avbd-demo3d\build-viewer-bridge-nmake\avbd_headless_server.exe --scene "Pyramid" --port 8765 --tick-rate 60
```

The browser viewer uses the same bridge URL and can control the headless server with scene load, pause/play, step, and reset commands.

## Run The GPU Headless Path

For the current playable GPU checkpoint, use the Dawn/WebGPU headless server with the resident async contact backend and binary snapshots:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run server:gpu
```

The launcher checks `127.0.0.1:8765` before using it. If that port is busy, it scans for the next available port and prints the matching viewer URL, for example `http://127.0.0.1:5173/?ws=...`.

In a second terminal, run the viewer:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run dev -- --port 5173
```

Useful launcher options:

```powershell
npm run server:gpu -- --scene "Sphere Pour 5000 on Cylinders"
npm run server:gpu -- --port 9000 --strict-port
npm run server:gpu -- --check-only
```

The default GPU launcher uses:

- physics backend: `webgpu-contact-resident-async`
- snapshot mode: `binary`
- tick rate: `60`

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

Validate that the GPU headless launcher starts and emits binary snapshots:

```powershell
cd C:\code\avbd-demo3d\viewer
npm run smoke:server:gpu
```

The viewer retries the bridge connection automatically, so it is fine to open the browser before launching the native app.
