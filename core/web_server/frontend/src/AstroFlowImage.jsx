import { useContext, useEffect, useState } from "react";
import TileViewer from "./TileViewer.jsx";
import AstroFlowContext from "./AstroFlowContext.jsx";
import QualityLevelSelector from "./QualityLevelSelector.jsx";

export const AstroFlowImage = ({
    objectType,
    objectId,
    filter = undefined,
    showPolygons = false,
    size = "full",
}) => {
    const { validity } = useContext(AstroFlowContext);

    const [qualityLevel, setQualityLevel] = useState(null);
    const [availableLevels, setAvailableLevels] = useState([]);

    useEffect(() => {
        const url = filter !== undefined
            ? `/api/available_qualities/${objectType}/${objectId}/${filter}`
            : `/api/available_qualities/${objectType}/${objectId}`;
        fetch(url)
            .then((res) => res.ok ? res.json() : null)
            .then((data) => {
                if (!data) return;
                setQualityLevel(data.quality_level);
                setAvailableLevels(data.available || []);
            })
            .catch(() => {});
    }, [objectType, objectId, filter]);

    // Merge server-side validity with client-side context validity
    // (context updates more frequently via SSE).
    const objectValidity = validity[`${objectType}:${objectId}`] || {};
    const enrichedLevels = availableLevels.map(l => ({
        ...l,
        valid: objectValidity[l.quality_level] ?? l.valid,
    }));

    const thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

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

                <div style={{
                    position: 'absolute', top: 8, left: 8,
                    pointerEvents: 'auto',
                }}>
                    <QualityLevelSelector
                        selected={qualityLevel}
                        available={enrichedLevels}
                        onSelect={setQualityLevel}
                    />
                </div>
            </div>
        </div>
    );
};
