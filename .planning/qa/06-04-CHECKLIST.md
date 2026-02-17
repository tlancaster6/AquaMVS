# Phase 06-04: CLI QA Execution — Checklist

## Task 1: Export mesh to all formats with simplification

### Step 1: Locate input mesh
- [x] 1.1: Run `ls output/frame_000000/mesh_*.ply` to find a PLY mesh from LightGlue run
- [x] 1.2: Run `python -c "import open3d as o3d; m = o3d.io.read_triangle_mesh('output/frame_000000/mesh_frame_000000.ply'); print(len(m.triangles), 'triangles')"` to get original triangle count

### Step 2: Export to OBJ
- [x] 2.1: Run `aquamvs export-mesh output/frame_000000/mesh_frame_000000.ply --format obj`
- [x] 2.2: Verify output/frame_000000/mesh_frame_000000.obj exists and is non-zero size

### Step 3: Export to STL
- [x] 3.1: Run `aquamvs export-mesh output/frame_000000/mesh_frame_000000.ply --format stl`
- [x] 3.2: Verify output/frame_000000/mesh_frame_000000.stl exists and is non-zero size

### Step 4: Export to GLTF with simplification
- [x] 4.1: Choose a target face count roughly 50% of original (or 50000, whichever is smaller)
- [x] 4.2: Run `aquamvs export-mesh output/frame_000000/mesh_frame_000000.ply --format gltf --simplify TARGET`
- [x] 4.3: Verify output/frame_000000/mesh_frame_000000.gltf exists
- [x] 4.4: Run `python -c "import open3d as o3d; m = o3d.io.read_triangle_mesh('output/frame_000000/mesh_frame_000000.gltf'); print(len(m.triangles), 'triangles')"` and verify triangle count is approximately target

### Step 5: Test batch mode (optional if time allows)
- [x] 5.1: If multiple PLY meshes exist, run `aquamvs export-mesh --input-dir output/frame_000000/ --format obj`

### Step 6: Log issues
- [x] 6.1: Append any non-blocking issues to .planning/qa/issues-found.md

## Task 2: User review of exported meshes

- [x] 7.1: Open OBJ file in MeshLab or 3D viewer and verify surface looks correct with outward-pointing normals
- [x] 7.2: Open STL file and verify it's a valid solid with no holes and correct normals
- [x] 7.3: Open GLTF file and verify simplification didn't destroy surface shape and reduced triangle count is visible
- [x] 7.4: Compare file sizes: STL should be largest (binary triangles), GLTF smallest (simplified)
- [x] 7.5: Signal completion by typing "approved"
