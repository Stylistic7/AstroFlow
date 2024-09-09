import React, { useEffect, useRef, useState } from 'react';
import './ImageWithOverlay.css';

const ImageWithOverlay = ({ src, alt, scale = 1 }) => {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const imageRef = useRef(null);

  useEffect(() => {
    // When the image loads, retrieve its natural width and height
    const handleImageLoad = () => {
      const { naturalWidth, naturalHeight } = imageRef.current;
      setImageSize({ width: naturalWidth, height: naturalHeight });
    };

    const imgElement = imageRef.current;
    if (imgElement.complete) {
      // If the image was cached and already loaded
      handleImageLoad();
    } else {
      imgElement.addEventListener('load', handleImageLoad);
    }

    return () => {
      if (imgElement) {
        imgElement.removeEventListener('load', handleImageLoad);
      }
    };
  }, [src]);

  const { width, height } = imageSize;
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;

  return (
    <div
      className="image-container"
      style={{
        width: `${scaledWidth}px`,
        height: `${scaledHeight}px`,
      }}
    >
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        className="background-image"
        style={{
          width: '100%',
          height: '100%',
        }}
      />
      {width > 0 && height > 0 && (
        <svg className="rule-of-thirds-overlay" viewBox={`0 0 ${width} ${height}`}>
          <line x1="0" y1={height / 3} x2={width} y2={height / 3} stroke="white" strokeWidth="0.5" />
          <line x1="0" y1={(height / 3) * 2} x2={width} y2={(height / 3) * 2} stroke="white" strokeWidth="0.5" />
          <line x1={width / 3} y1="0" x2={width / 3} y2={height} stroke="white" strokeWidth="0.5" />
          <line x1={(width / 3) * 2} y1="0" x2={(width / 3) * 2} y2={height} stroke="white" strokeWidth="0.5" />
        </svg>
      )}
    </div>
  );
};

export default ImageWithOverlay;
