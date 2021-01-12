# TP7 : B-spline surfaces

```bash
cd geonum/
git pull
```
or, if you don't have the local repo
```bash
git clone https://gricad-gitlab.univ-grenoble-alpes.fr/geonum/geonum.git
cd geonum/
```
Then
```bash
. exportPath.sh
```
Echo the `$PYTHONPATH` to verify it's been correctly set.
```bash
echo $PYTHONPATH
```
For the viewer to function properly, **python scripts need to be executed from the root dir**.
```bash
# test TP7
python TP7/tp7.py
```

---

## Functions to modify
* `DeBoorSurf` : recursively implement De Boor's algorithm for surfaces.
* `main part` : for each patch of the B-spline surface, evaluate surface points by calling `DeBoorSurf` in a double loop.

## ToDo
1. Implement evaluation of B-spline surfaces. Test with the provided datasets (`simple.bspline` and `torus.bspline`).
1. Modify the knot vectors for the simple dataset. Experiment with various configurations. How does the surface change?
1. [**Bonus**] NURBS surfaces can be used to represent the unit sphere, the same way we used NURBS curves to represent the unit circle in `TP3`. Modify your implementation of B-spline surfaces to compute NURBS surfaces. Test with the hemisphere (`hemi.nurbs`) and the modified torus (`torus.nurbs`).  
Note: full sphere control points, weights, and knots can be found in [Representing a Circle or a Sphere with NURBS](https://www.geometrictools.com/Documentation/NURBSCircleSphere.pdf) by David Eberly.
