import { useContext, useEffect, useState } from "react";
import TileViewer from "./TileViewer.jsx";
import AstroFlowContext from "./AstroFlowContext.jsx";
import QualityLevelSelector from "./QualityLevelSelector.jsx";

/**
 * Hook to fetch available quality levels for an object.
 * Returns [qualityLevel, setQualityLevel, enrichedLevels].
 */
export function useQualityLevels(objectType, objectId, filter) {
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
            .catch((e) => console.warn(e));
    }, [objectType, objectId, filter]);

    const objectValidity = validity[`${objectType}:${objectId}`] || {};
    const enrichedLevels = availableLevels.map(l => ({
        ...l,
        valid: objectValidity[l.quality_level] ?? l.valid,
    }));

    return [qualityLevel, setQualityLevel, enrichedLevels];
}

export const AstroFlowImage = ({
    objectType,
    objectId,
    filter = undefined,
    showPolygons = false,
    showPanelNumbers = false,
    size = "full",
    showQualitySelector = true,
    qualityLevel: qualityLevelProp = undefined,
    onQualityChange = undefined,
}) => {
    // Use controlled props if provided, otherwise manage internally.
    const [internalLevel, setInternalLevel, enrichedLevels] =
        useQualityLevels(objectType, objectId, filter);

    const qualityLevel = qualityLevelProp !== undefined ? qualityLevelProp : internalLevel;
    const setQualityLevel = onQualityChange || setInternalLevel;

    const thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
                {size === "full" ? (
                    <TileViewer
                        key={`${objectType}:${objectId}:${filter}:${qualityLevel}`}
                        objectType={objectType}
                        imageHash={objectId}
                        filter={filter}
                        qualityLevel={qualityLevel}
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

                {showQualitySelector && (
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
                )}
            </div>
        </div>
    );
};
