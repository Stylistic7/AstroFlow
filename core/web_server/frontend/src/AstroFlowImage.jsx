import React, { useEffect, useState } from "react";
import TileViewer from "./TileViewer.jsx";

function getBestQuality(metadata) {
    const qualityOrder = ['subframe', 'master', 'post_processed'];
    const availableLevels = Object.keys(metadata.levels || {});

    // Find the highest-quality level that exists in `metadata.levels`
    for (let i = qualityOrder.length - 1; i >= 0; i--) {
        if (availableLevels.includes(qualityOrder[i])) {
            return qualityOrder[i];
        }
    }

    return null; // or throw an error if no valid level is found
}

export const AstroFlowImage = ({
    objectType,
    objectId,
    filter,
    size = "full",
    forceLevel,
}) => {
    const [metadata, setMetadata] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch(`/metadata/${objectType}/${objectId}/${filter}`)
            .then((res) => res.json())
            .then(setMetadata)
            .catch((e) => setError("Image metadata not available"));
    }, [objectType, objectId, filter]);

    if (error) return <div>{error}</div>;
    if (!metadata) return <div>Loading...</div>;

    const level = forceLevel || getBestQuality(metadata);
    const info = metadata.levels[level];

    if (!info) return <div>No available image at level {level}</div>;

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

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ flexGrow: 1, position: 'relative', overflow: 'hidden' }}>
                {size === "full" ? (
                    <TileViewer objectType={objectType} imageHash={objectId} filter={filter} />
                ) : (
                    <img
                        src={size === "thumbnail" ? info.thumbnail_url : info.image_url}
                        alt={`${objectType} ${level}`}
                        style={{ width: "100%", height: "auto" }}
                    />
                )}
                <div style={labelStyle}>Level: {level}</div>
            </div>
        </div>
    );
};
