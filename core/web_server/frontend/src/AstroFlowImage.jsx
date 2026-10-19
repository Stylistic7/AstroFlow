import TileViewer from "./tile_viewer/TileViewerV2.jsx";


/**
 * Thin wrapper around TileViewer (full-size) or a thumbnail <img>
 * (any other `size`). The caller picks the full-size view via
 * (pipeline, finish) — passed straight through to TileViewer's URL
 * construction. The thumbnail variant needs no view: it is the object's
 * own per-type thumbnail.
 */
export const AstroFlowImage = ({
    objectType,
    objectId,
    showPolygons = false,
    showPanelNumbers = false,
    showDepth = false,
    size = "full",
    pipeline = undefined,
    finish = undefined,
    hdrEnabled = false,
    displayWhite = 1.0,
}) => (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
            {size === "full" ? (
                // Key intentionally omits the view (pipeline/finish).
                // Changing finish shouldn't tear down the canvas —
                // TileViewer handles the swap internally and preserves
                // zoom/pan.
                <TileViewer
                    key={`${objectType}:${objectId}`}
                    objectType={objectType}
                    imageHash={objectId}
                    pipeline={pipeline}
                    finish={finish}
                    showPolygons={showPolygons}
                    showPanelNumbers={showPanelNumbers}
                    showDepth={showDepth}
                    hdrEnabled={hdrEnabled}
                    displayWhite={displayWhite}
                />
            ) : (
                <img
                    src={`/thumbnail/${objectType}/${objectId}`}
                    alt={`${objectType} ${objectId}`}
                    style={{ width: "100%", height: "auto" }}
                />
            )}
        </div>
    </div>
);
