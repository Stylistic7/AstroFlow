import { useContext, useEffect, useState } from "react";
import TileViewer from "./TileViewer.jsx";
import AstroFlowContext from "./AstroFlowContext.jsx";

const qualityLabels = {
    "Subframe": "Single frame",
    "DownscaledMaster": "Downscaled",
    "Master": "Stacked",
    "PostProcessed": "Processed",
};

const qualityColors = {
    "Subframe": "rgba(255, 170, 50, 0.85)",
    "DownscaledMaster": "rgba(180, 130, 255, 0.85)",
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
    const { validity } = useContext(AstroFlowContext);

    const [qualityLevel, setQualityLevel] = useState(null);

    useEffect(() => {
        const url = filter !== undefined
            ? `/api/best_quality/${objectType}/${objectId}/${filter}`
            : `/api/best_quality/${objectType}/${objectId}`;
        fetch(url)
            .then((res) => res.ok ? res.json() : null)
            .then((data) => setQualityLevel(data?.quality_level ?? null))
            .catch(() => {});
    }, [objectType, objectId, filter]);

    let thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    const label = qualityLabels[qualityLevel];
    const objectValidity = validity[`${objectType}:${objectId}`];
    const isStale = qualityLevel && objectValidity && objectValidity[qualityLevel] === false;

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
                            background: isStale ? 'rgba(255, 160, 50, 0.85)' : qualityColors[qualityLevel],
                            color: 'white',
                            padding: '2px 8px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontWeight: 500,
                            width: 'fit-content',
                        }}>
                            {isStale ? `${label} (stale)` : label}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
