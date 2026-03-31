import { useContext, useEffect, useState } from "react";
import TileViewer from "./TileViewer.jsx";
import AstroFlowContext from "./AstroFlowContext.jsx";

const qualityLabels = {
    "Subframe": "Single frame",
    "Master": "Stacked",
    "PostProcessed": "Processed",
};

const qualityColors = {
    "Subframe": "rgba(255, 170, 50, 0.85)",
    "Master": "rgba(100, 160, 255, 0.85)",
    "PostProcessed": "rgba(80, 200, 120, 0.85)",
};

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

    const [qualityLevel, setQualityLevel] = useState(null);

    useEffect(() => {
        const url = filter !== undefined
            ? `/api/metadata/${objectType}/${objectId}/${filter}`
            : `/api/metadata/${objectType}/${objectId}`;
        fetch(url)
            .then((res) => res.ok ? res.json() : null)
            .then((meta) => setQualityLevel(meta?.quality_level ?? null))
            .catch(() => {});
    }, [objectType, objectId, filter]);

    let thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    const label = qualityLabels[qualityLevel];

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

                {/* Quality level + staleness indicators */}
                <div style={{
                    position: 'absolute', top: 8, left: 8,
                    display: 'flex', flexDirection: 'column', gap: 4,
                    pointerEvents: 'none',
                }}>
                    {label && (
                        <div style={{
                            background: qualityColors[qualityLevel],
                            color: 'white',
                            padding: '2px 8px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 500,
                            width: 'fit-content',
                        }}>
                            {label}
                        </div>
                    )}
                    {isStale && (
                        <div style={{
                            background: 'rgba(255, 100, 100, 0.85)',
                            color: 'white',
                            padding: '2px 8px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 500,
                            width: 'fit-content',
                        }}>
                            Reprocessing…
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
