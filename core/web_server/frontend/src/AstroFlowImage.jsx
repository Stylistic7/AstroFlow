import { useContext } from "react";
import TileViewer from "./TileViewer.jsx";
import AstroFlowContext from "./AstroFlowContext.jsx";

export const AstroFlowImage = ({
    objectType,
    objectId,
    filter = undefined,
    showPolygons = false,
    size = "full",
}) => {
    const { invalidated } = useContext(AstroFlowContext);
    const isStale = (invalidated || []).some(
        ([type, id]) => type === objectType && id === objectId
    );

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
                {isStale && (
                    <div style={{
                        position: 'absolute', top: 8, right: 8,
                        background: 'rgba(255, 100, 100, 0.85)',
                        color: 'white', padding: '2px 8px',
                        borderRadius: 4, fontSize: 12,
                        pointerEvents: 'none',
                    }}>
                        Reprocessing...
                    </div>
                )}
            </div>
        </div>
    );
};
