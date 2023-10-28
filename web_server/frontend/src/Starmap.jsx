import React, { useState } from "react";

import './Starmap.css';

export default function Starmap({stacks, targets, mosaics}) {
	const [pos, setPos] = useState({ x: 0, y: 0, scale: 1 });

  const onScroll = (e) => {
    const delta = e.deltaY * -0.01;
    const newScale = pos.scale + delta;

    const ratio = 1 - newScale / pos.scale;

    setPos({
      scale: newScale,
      x: pos.x + (e.clientX - pos.x) * ratio,
      y: pos.y + (e.clientY - pos.y) * ratio
    });
  };

  // TODO: This probably isn't the right way to do this in React.
	var svg = document.getElementById("starmap_svg");

	for (const [key, target] of Object.entries(targets)) {
		console.log('target', target);
		console.log('target.footprint', target['footprint']);

		var polygon = document.createElementNS(
			"http://www.w3.org/2000/svg", "polygon");
		polygon.classList.add("starmap_polygon");
		svg.appendChild(polygon);
		polygon.addEventListener('click', () => console.log(target, 'clicked'));

		for (const point of target.footprint) {
		  var svg_point = svg.createSVGPoint();
		  svg_point.x = 360 - (point[0] + 180) % 360;
		  svg_point.y = 90 - point[1];
		  // console.log('point', point[0], point[1]);
		  polygon.points.appendItem(svg_point);
		}
	}

	return (
		<div id="starmap_container"
				 onWheelCapture={onScroll} 
				 style={{
           transformOrigin: "0 0",
           transform: `translate(${pos.x}px, ${pos.y}px) scale(${pos.scale})`
         }} >
			<img id="starmap_img" src="/static/images/starmap_2020_4k.png" />
			<img id="constellation_figures_img" src="/static/images/constellation_figures_4k.png" />
			<svg id="starmap_svg" viewBox="0 0 360 180"></svg>
		</div>
	);
}