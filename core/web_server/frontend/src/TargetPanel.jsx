import React, { useContext } from "react";
import { useParams } from "react-router-dom";
import { Box } from "@mui/material";

import StackThumbnail from "./StackThumbnail.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import SuggestionBadge from "./suggestions/SuggestionBadge.jsx";
import QuarantineToggle from "./QuarantineToggle.jsx";
import MosaicMembershipToggle from "./MosaicMembershipToggle.jsx";
import ShoppingCartMenu from "./ShoppingCartMenu.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import { PipelineSection } from './MosaicDashboard.jsx';
import { useFinishes } from './useVersions.js';
import { buildUrl } from './urls.js';


export default function TargetPanel() {
	const { stacks, targets, isQuarantined, computeGraphStatus } = useContext(AstroFlowContext);

	const params = useParams();

	const target = targets[params.target_id];

	const hierarchy = [];
	if (params.mosaic_id) hierarchy.push({ type: 'mosaic', id: params.mosaic_id });
	hierarchy.push({ type: 'target', id: params.target_id });

	// Pipeline output sections — same per-pipeline finish list the mosaic
	// dashboard shows. Empty (no sections) for a target no pipeline has
	// produced finishes for.
	const finishes = useFinishes("target", params.target_id, computeGraphStatus);
	const pipelines = (finishes?.pipelines || [])
		.filter((p) => p.finishes.length > 0);

	// No page-level hero here either: each PipelineSection renders its own
	// canonical preview, deep-linked into that pipeline's viewer.

	// Drilldown links are hierarchy-only: the stack page picks its own
	// default view.
	const handleStackClick = (stack_id) => () => {
		window.open(buildUrl({ hierarchy: [...hierarchy, { type: 'stack', id: stack_id }] }), '_blank');
	};

	return (
		<div>
			<div style={{ display: "flex", alignItems: "center", marginLeft: "2em", marginBottom: "0.5em" }}>
				<ShoppingCartMenu objectType="target" objectId={params.target_id} />
				<QuarantineToggle objectType="target" objectId={params.target_id} />
				<MosaicMembershipToggle targetId={params.target_id} />
				<SuggestionBadge objectType="target" objectId={params.target_id} />
			</div>
			<div style={{
					display: "grid",
					gridTemplateColumns: "repeat(4, 1fr)",
					maxHeight: "80vh",
					gap: "1em",
					marginLeft: "2em",
					marginRight: "2em",
					opacity: isQuarantined("target", params.target_id) ? 0.5 : 1,
				}}>
					{target?.stack_ids?.map((stack_id) => (
						<StackThumbnail
							key={"cell_" + stack_id}
							stacks={stacks}
							stack_id={stack_id}
							handleClick={handleStackClick(stack_id)}
						>
							<DecisionBadge objectType="stack" objectId={stack_id} />
							<ConcernsBadge objectType="stack" objectId={stack_id} />
						</StackThumbnail>
					))}
			</div>
			{pipelines.length > 0 && (
				<Box sx={{ mx: "2em", mt: "1.5em" }}>
					{pipelines.map((p) => (
						<PipelineSection
							key={p.name}
							objectType="target"
							objectId={params.target_id}
							pipeline={p}
						/>
					))}
				</Box>
			)}
		</div>
	);
}
