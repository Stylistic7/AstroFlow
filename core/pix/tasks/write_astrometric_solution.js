#engine v8

/*
 * WriteAstrometricSolution.js
 *
 * Take a pre-computed CD-matrix WCS (from solve-field or similar
 * external solver) and emit PixInsight's canonical astrometric
 * solution onto the active view: refreshed FITS WCS keywords plus the
 * PCL:AstrometricSolution:* XISF properties that SPCC, MGC, and
 * StarAlignment consume.
 *
 * This is the same persistence flow ImageSolver itself uses after it
 * has a solution; we just hand it the answer directly instead of
 * running the pair-RANSAC step that fails on dense Milky Way fields.
 *
 * Parameters (via -p flags):
 *   view_id   The PixInsight view ID of the master image.
 *   crval1    RA of reference pixel, degrees.
 *   crval2    Dec of reference pixel, degrees.
 *   crpix1    X pixel of reference (FITS 1-based).
 *   crpix2    Y pixel of reference (FITS 1-based).
 *   cd1_1, cd1_2, cd2_1, cd2_2   CD matrix elements, deg/pixel.
 */

// WCSmetadata.jsh expects DataType_*, PropertyType_*, PropertyAttribute_*,
// and the RBFType / UndoFlag enums to be defined when its bodies run, so
// the underlying pjsr includes have to come first. Order matches what
// ImageSolver.js uses internally.
#include <pjsr/DataType.jsh>
#include <pjsr/PropertyType.jsh>
#include <pjsr/PropertyAttribute.jsh>
#include <pjsr/UndoFlag.jsh>
#include <pjsr/RBFType.jsh>
#include "/Applications/PixInsight/src/scripts/AdP/WCSmetadata.jsh"

function pget(name, defaultValue) {
   if (Parameters.has(name))
      return Parameters.get(name);
   return defaultValue;
}

function pgetf(name, defaultValue) {
   if (Parameters.has(name))
      return parseFloat(Parameters.get(name));
   return defaultValue;
}

var view_id = pget("view_id", "");
if (!view_id)
   throw new Error("WriteAstrometricSolution: view_id parameter required");

var view = View.viewById(view_id);
if (view.isNull)
   throw new Error("WriteAstrometricSolution: view not found: " + view_id);
var win = view.window;

var crval1 = pgetf("crval1", null);
var crval2 = pgetf("crval2", null);
var crpix1 = pgetf("crpix1", null);
var crpix2 = pgetf("crpix2", null);
var cd1_1  = pgetf("cd1_1",  null);
var cd1_2  = pgetf("cd1_2",  null);
var cd2_1  = pgetf("cd2_1",  null);
var cd2_2  = pgetf("cd2_2",  null);

if (crval1 === null || crval2 === null || crpix1 === null || crpix2 === null ||
    cd1_1 === null || cd1_2 === null || cd2_1 === null || cd2_2 === null)
   throw new Error("WriteAstrometricSolution: missing required parameter(s)");

console.writeln("WriteAstrometricSolution: view=" + view_id +
                " RA=" + crval1.toFixed(4) +
                " Dec=" + crval2.toFixed(4));

// --- 1. Replace any existing WCS-related FITS keywords with the new ones.
var WCS_KEYS = ["CTYPE1","CTYPE2","CRVAL1","CRVAL2","CRPIX1","CRPIX2",
                "CD1_1","CD1_2","CD2_1","CD2_2","CUNIT1","CUNIT2",
                "LONPOLE","LATPOLE","PV1_1","PV1_2","EQUINOX","RADESYS",
                "CDELT1","CDELT2","CROTA1","CROTA2"];
var kw = win.keywords.filter(function(k) {
   return WCS_KEYS.indexOf(k.name) === -1;
});
kw.push(new FITSKeyword("CTYPE1", "'RA---TAN'", "RA gnomonic projection"));
kw.push(new FITSKeyword("CTYPE2", "'DEC--TAN'", "Dec gnomonic projection"));
kw.push(new FITSKeyword("CRVAL1", crval1.toFixed(10), "RA of reference pixel (deg)"));
kw.push(new FITSKeyword("CRVAL2", crval2.toFixed(10), "Dec of reference pixel (deg)"));
kw.push(new FITSKeyword("CRPIX1", crpix1.toFixed(6), "X pixel of reference"));
kw.push(new FITSKeyword("CRPIX2", crpix2.toFixed(6), "Y pixel of reference"));
kw.push(new FITSKeyword("CD1_1",  cd1_1.toExponential(15), "CD matrix"));
kw.push(new FITSKeyword("CD1_2",  cd1_2.toExponential(15), "CD matrix"));
kw.push(new FITSKeyword("CD2_1",  cd2_1.toExponential(15), "CD matrix"));
kw.push(new FITSKeyword("CD2_2",  cd2_2.toExponential(15), "CD matrix"));
kw.push(new FITSKeyword("CUNIT1", "'deg'", ""));
kw.push(new FITSKeyword("CUNIT2", "'deg'", ""));
kw.push(new FITSKeyword("EQUINOX", "2000.0", "ICRS J2000.0"));
kw.push(new FITSKeyword("RADESYS", "'ICRS'", "Reference frame"));
win.keywords = kw;

// --- 2. Discard any solution the view already carries. ExtractMetadata
// prefers an existing PCL:AstrometricSolution:LinearTransformationMatrix
// property over the keywords (WCSmetadata.jsh, `fromFITS`), and PI's FITS
// loader builds one on load from WCS keywords embedded in the file — so
// on such frames the keywords written above were silently ignored and
// SPFC ran on the loader's solution (mirrored for half the camera
// orientations; found 2026-09-07).
var stale = view.properties.filter(function(id) {
   return id.indexOf("PCL:AstrometricSolution:") === 0 ||
          id.indexOf("Transformation_ImageToProjection") === 0;
});
view.beginProcess(UndoFlag_AstrometricSolution);
stale.forEach(function(id) { view.deleteProperty(id); });
if (stale.length > 0)
   console.writeln("WriteAstrometricSolution: discarded " + stale.length +
                   " pre-existing astrometric solution properties");

// --- 3. Hand ExtractMetadata the solution as PI's own properties, not as
// keywords. A keyword-sourced solution goes through a bottom-up guess
// (`fromFITS`: the sign of a CDELT2 estimated from the CD matrix's
// rotation branch decides whether the rows are flipped) and that guess
// is invariant to how the WCS is written, so for det>0 matrices it
// mirrors the solution for half the camera orientations (every +90°
// frame; 2026-09-07). A property-sourced solution is used verbatim:
// image->gnomonic matrix = CD, reference pixel = CRPIX in PI's 0-based
// image coordinates (rows in file order, no flip — verified on -90°/+90°
// frames against solve-field's own catalogue positions).
var props = PropertyAttribute_Storable | PropertyAttribute_Permanent;
view.setPropertyValue("PCL:AstrometricSolution:LinearTransformationMatrix",
                      new Matrix([cd1_1, cd1_2, cd2_1, cd2_2], 2, 2), PropertyType_F64Matrix, props);
view.setPropertyValue("PCL:AstrometricSolution:ReferenceImageCoordinates",
                      new Vector([crpix1 - 0.5, crpix2 - 0.5]), PropertyType_F64Vector, props);
view.setPropertyValue("PCL:AstrometricSolution:ReferenceCelestialCoordinates",
                      new Vector([crval1, crval2]), PropertyType_F64Vector, props);
view.endProcess();

// --- 4. Drive PI's own ImageMetadata class to extract that solution and
// emit the canonical representation: refreshed FITS keywords
// (UpdateBasicKeywords + UpdateWCSKeywords) plus the full set of
// PCL:AstrometricSolution:* XISF properties that SPCC/MGC consume.
var metadata = new ImageMetadata("WriteAstrometricSolution");
metadata.ExtractMetadata(win);

view.beginProcess(UndoFlag_Keywords | UndoFlag_AstrometricSolution);
metadata.SaveKeywords(win, false);
metadata.SaveProperties(win, "AstroFlow", "external (solve-field)");
win.regenerateAstrometricSolution();
view.endProcess();

console.writeln("WriteAstrometricSolution: done.");
