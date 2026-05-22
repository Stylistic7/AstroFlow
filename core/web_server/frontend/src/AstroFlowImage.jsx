import TileViewer from "./TileViewer.jsx";


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
    size = "full",
    pipeline = undefined,
    display = undefined,
}) => {
    const thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
                {size === "full" ? (
                    <TileViewer
                        key={`${objectType}:${objectId}:${filter}:${pipeline}:${display}`}
                        objectType={objectType}
                        imageHash={objectId}
                        filter={filter}
                        pipeline={pipeline}
                        display={display}
                        showPolygons={showPolygons}
                        showPanelNumbers={showPanelNumbers}
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
