import React, { useEffect, useRef, useState } from 'react';
import './ImageWithOverlay.css';

const ImageWithOverlay = ({ src, alt, scale = 1, overlay }) => {
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
      {scaledWidth > 0 && scaledHeight > 0 && overlay && (
        <div className="overlay-container">
          {overlay({ scaledWidth, scaledHeight })}
        </div>
      )}
    </div>
  );
};

export default ImageWithOverlay;
