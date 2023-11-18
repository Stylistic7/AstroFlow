import React, { useState, useRef, useEffect, useReducer } from "react";
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function MosaicPanel({ mosaic_id, mosaics, targets, stacks, handleClick }) {
	return (
		<div style={{margin: "2px"}}>
			<table width="100%">
				<thead height="0"></thead>
				<tbody>
					{mosaics[mosaic_id].target_ids.map((target_id) => (
						<TargetPanel
						  id={"TargetPanel_" + target_id}
						  key={target_id}
							target={targets[target_id]}
							stacks={stacks}
							handleClick={handleClick}
						/>
					))}
				</tbody>
			</table>
		</div>
	);
}
