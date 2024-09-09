import React from 'react';

const CircleSlashIcon = ({ IconComponent, iconProps = {}, x, y, width, height }) => {
  return (
    <svg x={x} y={y} width={width} height={height} viewBox={`0 0 ${width} ${height}`} xmlns="http://www.w3.org/2000/svg">
      {/* Render the icon SVG within this SVG */}
      <IconComponent {...iconProps} />

      {/* Slash overlay */}
      <line
        x1="0"
        y1={height}
        x2={width}
        y2="0"
        stroke="white"
        strokeWidth="0.0005"
      />
    </svg>
  );
};

export default CircleSlashIcon;
