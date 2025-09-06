import React, { useState, useRef, useEffect, useReducer, useContext } from "react";
import { useNavigate } from "react-router-dom";
import "./Starmap.css";
import { ReactPhotoSphereViewer } from 'react-photo-sphere-viewer';
import { MarkersPlugin } from '@photo-sphere-viewer/markers-plugin';
import "@photo-sphere-viewer/markers-plugin/index.css";
import AstroFlowContext from './AstroFlowContext.jsx';
import MosaicTabs from "./MosaicTabs.jsx";

export default function Starmap() {
	const { constellationPolylines, targets, mosaics } = useContext(AstroFlowContext);
	const navigate = useNavigate();
	const [instance, setInstance] = useState(undefined);

	const handleClick = (data: events.ClickEvent & { type: "click"; }) => {
		let elements = document.elementsFromPoint(data.data.clientX, data.data.clientY);
		let matching_items = [];
		elements.forEach((elt, i) => {
			let id = elt.getAttribute("id");
			// TODO: This is ugly. Is there a better way to attach arbitrary
			// data to the SVG elements used in markers?
			if (id && id.startsWith("psv-marker-")) {
				id = id.substring("psv-marker-".length);
				matching_items.push(id);
			}
		});

		navigate(`/disambiguation/${matching_items.join(";")}`);
	};

	const copyright = <span id="starmap_copyright">
		&copy; ESA/Gaia/DPAC; CC BY-SA 3.0 IGO. Acknowledgement: A.
		Moitinho.
	</span>

	const plugins = [
		[MarkersPlugin, {
			markers: [],
			clickEventOnMarker: true
		}],
	]

	useEffect(() => {
		if (!constellationPolylines || !targets || !mosaics)
			return;

		const markersPlugs = instance?.getPlugin(MarkersPlugin);
		if (!markersPlugs)
			return;

		markersPlugs.clearMarkers();

		constellationPolylines.map((polyline, index) => {
			markersPlugs.addMarker({
				id: "polyline_" + index,
				polyline: polyline,
				style: {
					stroke: 'white',
					strokeOpacity: '0.5',
					strokeWidth: '1px',
				},
			});
		});

		for (let [id, target] of Object.entries(targets)) {
			if (!target.projected_footprints) continue;

			markersPlugs.addMarker({
				id: "target," + id,
				polygon: target.projected_footprints['galactic']['coords'],
				style: {
					stroke: 'rgba(0, 0, 255, 0.7)',
					strokeOpacity: '0.7',
					strokeWidth: '4px',
					fill: 'black',
					fillOpacity: '0.2',
				}
			});
		};

		for (let [id, mosaic] of Object.entries(mosaics)) {
			if (!mosaic.projected_footprints) continue;

			markersPlugs.addMarker({
				id: "mosaic," + id,
				polygon: mosaic.projected_footprints['galactic']['coords'],
				style: {
					stroke: 'rgba(255, 0, 0, 0.8)',
					strokeOpacity: '0.8',
					strokeWidth: '4px',
					fill: 'black',
					fillOpacity: '0.2',
				}
			});
		};
	}, [instance, constellationPolylines, targets, mosaics]);


	const handleReady = (instance) => {
		setInstance(instance);
	}; // end handleReady

	return <ReactPhotoSphereViewer
		src="/static/images/ESA_Gaia_DR2_AllSky_Brightness_Colour_Cartesian_4000x2000.png"
		bottom="0"
		height="90vh"
		width="100%"
		plugins={plugins}
		minFov={10}
		maxFov={140}
		onReady={handleReady}
		onClick={handleClick} >
	</ReactPhotoSphereViewer >;
}
