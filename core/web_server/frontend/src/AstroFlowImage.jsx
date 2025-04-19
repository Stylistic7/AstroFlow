import TileViewer from "./TileViewer.jsx";

export const AstroFlowImage = ({
    objectType,
    objectId,
    filter = undefined,
    showPolygons = false,
    size = "full",
}) => {
    const labelStyle = {
        position: "absolute",
        top: 4,
        right: 4,
        background: "rgba(0,0,0,0.5)",
        color: "white",
        padding: "2px 6px",
        borderRadius: 4,
        fontSize: 12,
    };

    let thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
                {size === "full" ? (
                    <TileViewer
                        objectType={objectType}
                        imageHash={objectId}
                        filter={filter}
                        showPolygons={showPolygons}
                    />
                ) : (
                    <img
                        src={thumbnail_url}
                        alt={`${objectType} ${objectId}`}
                        style={{ width: "100%", height: "auto" }}
                    />
                )}
                {/* <div style={labelStyle}>Level: {level}</div> */}
            </div>
        </div>
    );
};
