import React, { useContext } from "react";

import AstroFlowContext from './AstroFlowContext.jsx';

// Zone.js is required before importing the Workflow Graph
import 'zone.js';

// You can import types from the package as you normally would
import { WorkflowGraphProps, Logger, RankerAlgorithim, LayoutOptions, RankDirection, RankAlignment } from '@google/workflow-graph';

// Import and register the google-workflow-graph web component
import '@google/workflow-graph/web-component/registerWorkflowGraphWebComponent';
// import './registerWorkflowGraphWebComponent.js';

import './WorkflowGraph.css';

const NODE_VERTICAL_SPACING = 60;
const NODE_HORIZONTAL_SPACING = 40;

const logger: Logger = { logEvent: (event) => { console.log(event) } };

const DEFAULT_LAYOUT_OPTIONS: LayoutOptions = {
  rankDirection: RankDirection.TOP_TO_BOTTOM,
  rankAlignment: RankAlignment.NONE,
  ranker: RankerAlgorithim.NETWORK_SIMPLEX,
  edgeSeparation: 4,
  rankSeparation: NODE_VERTICAL_SPACING,
  nodeSeparation: NODE_HORIZONTAL_SPACING,
};

export default function WorkflowGraph({ theme }) {
  const { computeGraph } = useContext(AstroFlowContext);

  const graphRef = React.useRef();

  React.useEffect(() => {

    // TODO: Better typing for the ref
    const graphComponent = graphRef.current;

    const onSelectedNodeChange = (event) => {
      console.log(event);
    }

    if (graphComponent) {
      const props = {
        enableMinimap: true,
        enableToolbar: true,
        expandedMode: false,
        selectedNode: null,
        zoom: 1,
        loading: false,
        dagSpec: { skeleton: computeGraph.skeleton, meta: computeGraph.state },
        optimizeForOrm: false,
        followNode: null,
        layout: DEFAULT_LAYOUT_OPTIONS,
        hoveredEdge: undefined,
        logger: logger,
      }

      Object.assign(graphComponent, props);
      graphComponent.addEventListener("selectedNodeChange", onSelectedNodeChange);
    }

    return () => {
      graphComponent?.removeEventListener("selectedNodeChange", onSelectedNodeChange);

    }
  }, [])

  return (
    <div className="WorkflowGraph">
      <div className="example-wrapper">
        <google-workflow-graph ref={graphRef}></google-workflow-graph>
        <workflow-graph-color-theme-loader />
      </div>
    </div>
  );
}
