import React, { useContext } from "react";
import { useParams } from "react-router-dom";
import { Box, Typography } from "@mui/material";

import StackThumbnail from "./StackThumbnail.jsx";
import DecisionBadge from "./DecisionBadge.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import SuggestionBadge from "./suggestions/SuggestionBadge.jsx";
import QuarantineToggle from "./QuarantineToggle.jsx";
import MosaicMembershipToggle from "./MosaicMembershipToggle.jsx";
import SharingMenu from "./SharingMenu.jsx";
import ShoppingCartMenu from "./ShoppingCartMenu.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import { PipelineSection } from './MosaicDashboard.jsx';
import { useFinishes } from './useVersions.js';
import { buildUrl } from './urls.js';


// A target is shown one PIPELINE at a time: each section lists the stacked
// masters that pipeline made (opening the stack viewer in that pipeline, so
// the stack page never guesses which pipeline's master to show) and the
// finishes applied to its outputs. Two pipelines (say WBPP and Siril
// stacking) are two sections on one page.
export default function TargetPanel() {
	const { stacks, targets, isQuarantined, computeGraphStatus } = useContext(AstroFlowContext);
	const params = useParams();
	const target = targets[params.target_id];

	const hierarchy = [];
	if (params.mosaic_id) hierarchy.push({ type: 'mosaic', id: params.mosaic_id });
	hierarchy.push({ type: 'target', id: params.target_id });

	const finishes = useFinishes("target", params.target_id, computeGraphStatus);
	const pipelines = finishes?.pipelines || [];

	const badges = (stack_id) => (
		<>
			<DecisionBadge objectType="stack" objectId={stack_id} />
			<ConcernsBadge objectType="stack" objectId={stack_id} />
		</>
	);

	return (
		<div style={{ opacity: isQuarantined("target", params.target_id) ? 0.5 : 1 }}>
			<div style={{ display: "flex", alignItems: "center", marginLeft: "2em", marginBottom: "0.5em" }}>
				<ShoppingCartMenu objectType="target" objectId={params.target_id} />
				<QuarantineToggle objectType="target" objectId={params.target_id} />
				<MosaicMembershipToggle targetId={params.target_id} />
				<SharingMenu objectType="target" objectId={params.target_id} />
				<SuggestionBadge objectType="target" objectId={params.target_id} />
			</div>
			<Box sx={{ mx: "2em", mt: "1em" }}>
				{pipelines.map((p) => (
					<PipelineSection
						key={p.name}
						objectType="target"
						objectId={params.target_id}
						hierarchy={hierarchy}
						pipeline={p}
						masterBadges={badges}
					/>
				))}
				{finishes && pipelines.length === 0 && (
					// Nothing has processed it yet: say so, and keep its stacks
					// reachable (their pages open in their own default view).
					<>
						<Typography sx={{ color: 'text.secondary', mb: 2 }}>
							No pipeline has processed this target yet. Add it to the cart to stack it.
						</Typography>
						<Box sx={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1em" }}>
							{target?.stack_ids?.map((stack_id) => (
								<StackThumbnail
									key={stack_id}
									stacks={stacks}
									stack_id={stack_id}
									handleClick={() => window.open(buildUrl({
										hierarchy: [...hierarchy, { type: 'stack', id: stack_id }] }), '_blank')}
								>
									{badges(stack_id)}
								</StackThumbnail>
							))}
						</Box>
					</>
				)}
			</Box>
		</div>
	);
}
