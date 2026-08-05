import TileViewer from "./tile_viewer/TileViewerV2.jsx";


/**
 * Thin wrapper around TileViewer (full-size) or a thumbnail <img>
 * (small-size). Caller controls which tap point is shown via
 * (pipeline, display) — passed straight through to TileViewer's URL
 * construction. For the `size="small"` thumbnail variant, no
 * (pipeline, display) is needed: a default-display thumbnail per
 * object type is fine.
 */
export const AstroFlowImage = ({
    objectType,
    objectId,
    filter = undefined,
    showPolygons = false,
    showPanelNumbers = false,
    showDepth = false,
    size = "full",
    pipeline = undefined,
    display = undefined,
    hdrEnabled = false,
    peakBrightness = 1.0,
    stretchMode = 'stf',
    onColorDetected = undefined,
}) => {
    const thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
                {size === "full" ? (
                    // Key intentionally omits the view triple
                    // (pipeline/display/filter). Changing tap shouldn't
                    // tear down the PIXI canvas — TileViewer handles
                    // the swap internally and preserves zoom/pan across
                    // the viewport re-create.
                    <TileViewer
                        key={`${objectType}:${objectId}`}
                        objectType={objectType}
                        imageHash={objectId}
                        filter={filter}
                        pipeline={pipeline}
                        display={display}
                        showPolygons={showPolygons}
                        showPanelNumbers={showPanelNumbers}
                        showDepth={showDepth}
                        hdrEnabled={hdrEnabled}
                        peakBrightness={peakBrightness}
                        stretchMode={stretchMode}
                        onColorDetected={onColorDetected}
                    />
                ) : (
                    <img
                        src={thumbnail_url}
                        alt={`${objectType} ${objectId}`}
                        style={{ width: "100%", height: "auto" }}
                    />
                )}
            </div>
        </div>
    );
};
