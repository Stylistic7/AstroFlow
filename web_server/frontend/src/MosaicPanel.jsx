import React, { useState, useRef, useEffect, useReducer } from "react";
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function MosaicPanel({ mosaic, mosaics, targets, stacks, zoomToElement }) {
	return (
		<div style={{margin: "2px"}}>
			{mosaics[mosaic].target_ids.map((target_id) => (
				<TargetPanel
				  id={"TargetPanel_" + target_id}
				  key={target_id}
					target={targets[target_id]}
					stacks={stacks}
					zoomToElement={zoomToElement}
					style={{border: '1px solid blue'}}
				/>
			))}
		</div>
	);
}
