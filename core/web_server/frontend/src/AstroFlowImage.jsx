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

const staleColor = "rgba(255, 160, 50, 0.85)";

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
    const [dropdownOpen, setDropdownOpen] = useState(false);

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

    const thumbnail_url = `/thumbnail/${objectType}/${objectId}${filter ? `/${filter}` : ''}`;

    const objectValidity = validity[`${objectType}:${objectId}`] || {};
    const isStale = qualityLevel && objectValidity[qualityLevel] === false;
    const label = qualityLabels[qualityLevel];
    const color = isStale ? staleColor : qualityColors[qualityLevel];

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

                {/* Quality level dropdown selector */}
                {label && (
                    <div style={{
                        position: 'absolute', top: 8, left: 8,
                        pointerEvents: 'auto',
                        userSelect: 'none',
                    }}>
                        <div
                            onClick={() => availableLevels.length > 1 && setDropdownOpen(v => !v)}
                            style={{
                                background: color,
                                color: 'white',
                                padding: '2px 10px',
                                borderRadius: 4,
                                fontSize: 11,
                                fontWeight: 500,
                                cursor: availableLevels.length > 1 ? 'pointer' : 'default',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 4,
                            }}
                        >
                            {isStale ? `${label} (stale)` : label}
                            {availableLevels.length > 1 && (
                                <span style={{ fontSize: 8 }}>{dropdownOpen ? '▲' : '▼'}</span>
                            )}
                        </div>

                        {dropdownOpen && availableLevels.length > 1 && (
                            <div style={{
                                marginTop: 2,
                                background: 'rgba(30, 30, 30, 0.95)',
                                borderRadius: 4,
                                overflow: 'hidden',
                                boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
                            }}>
                                {availableLevels.map(({ quality_level, valid }) => {
                                    const lvlLabel = qualityLabels[quality_level] || quality_level;
                                    const lvlStale = valid === false;
                                    const lvlColor = lvlStale ? staleColor : qualityColors[quality_level];
                                    const selected = quality_level === qualityLevel;
                                    return (
                                        <div
                                            key={quality_level}
                                            onClick={() => {
                                                setQualityLevel(quality_level);
                                                setDropdownOpen(false);
                                            }}
                                            style={{
                                                padding: '4px 10px',
                                                fontSize: 11,
                                                cursor: 'pointer',
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 6,
                                                background: selected ? 'rgba(255,255,255,0.1)' : 'transparent',
                                            }}
                                        >
                                            <span style={{
                                                width: 8, height: 8, borderRadius: '50%',
                                                background: lvlColor,
                                                display: 'inline-block',
                                                flexShrink: 0,
                                            }} />
                                            <span style={{
                                                color: 'white',
                                                fontWeight: selected ? 600 : 400,
                                            }}>
                                                {lvlStale ? `${lvlLabel} (stale)` : lvlLabel}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};
