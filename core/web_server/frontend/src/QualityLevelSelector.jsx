import { useState } from "react";

const qualityLabels = {
    "Subframe": "Subframe",
    "DownscaledMaster": "Downscaled",
    "Master": "Master",
    "PostProcessed": "Post-processed",
};

const qualityColors = {
    "Subframe": "rgba(255, 170, 50, 0.85)",
    "DownscaledMaster": "rgba(180, 130, 255, 0.85)",
    "Master": "rgba(100, 160, 255, 0.85)",
    "PostProcessed": "rgba(80, 200, 120, 0.85)",
};

const staleColor = "rgba(255, 160, 50, 0.85)";

/**
 * Dropdown selector for quality levels with validity indicators.
 *
 * Props:
 *   selected    — currently selected quality level string (e.g. "Master")
 *   available   — array of {quality_level, valid} from /api/available_qualities
 *   onSelect    — callback(quality_level) when the user picks a level
 */
export default function QualityLevelSelector({ selected, available, onSelect }) {
    const [open, setOpen] = useState(false);

    if (!selected || !available?.length) return null;

    const label = qualityLabels[selected] || selected;
    const isStale = available.find(l => l.quality_level === selected)?.valid === false;
    const color = isStale ? staleColor : qualityColors[selected];
    const hasChoices = available.length > 1;

    return (
        <div style={{ userSelect: "none" }}>
            <div
                onClick={() => hasChoices && setOpen(v => !v)}
                style={{
                    background: color,
                    color: "white",
                    padding: "2px 10px",
                    borderRadius: 4,
                    fontSize: 11,
                    fontWeight: 500,
                    cursor: hasChoices ? "pointer" : "default",
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                }}
            >
                {isStale ? `${label} (stale)` : label}
                {hasChoices && (
                    <span style={{ fontSize: 8 }}>{open ? "▲" : "▼"}</span>
                )}
            </div>

            {open && hasChoices && (
                <div style={{
                    marginTop: 2,
                    background: "rgba(30, 30, 30, 0.95)",
                    borderRadius: 4,
                    overflow: "hidden",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
                }}>
                    {available.map(({ quality_level, valid }) => {
                        const lvlLabel = qualityLabels[quality_level] || quality_level;
                        const lvlStale = valid === false;
                        const lvlColor = lvlStale ? staleColor : qualityColors[quality_level];
                        const isSelected = quality_level === selected;
                        return (
                            <div
                                key={quality_level}
                                onClick={() => { onSelect(quality_level); setOpen(false); }}
                                style={{
                                    padding: "4px 10px",
                                    fontSize: 11,
                                    cursor: "pointer",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 6,
                                    background: isSelected ? "rgba(255,255,255,0.1)" : "transparent",
                                }}
                            >
                                <span style={{
                                    width: 8, height: 8, borderRadius: "50%",
                                    background: lvlColor,
                                    display: "inline-block",
                                    flexShrink: 0,
                                }} />
                                <span style={{
                                    color: "white",
                                    fontWeight: isSelected ? 600 : 400,
                                }}>
                                    {lvlStale ? `${lvlLabel} (stale)` : lvlLabel}
                                </span>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
