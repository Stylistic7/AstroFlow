import React, { useContext, useRef, useState, useEffect } from "react";

import { NodeStatusDialog } from "./NodeStatusDialog.jsx";
import AstroFlowContext from "./AstroFlowContext.jsx";

// Zone.js is required before importing the Workflow Graph
import 'zone.js';

// You can import types from the package as you normally would
import { WorkflowGraphProps, Logger, RankerAlgorithim, LayoutOptions, RankDirection, RankAlignment } from '@google/workflow-graph';

// Import and register the google-workflow-graph web component
import '@google/workflow-graph/web-component/registerWorkflowGraphWebComponent';

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

export default function WorkflowGraph({ theme, url }) {
  const { computeGraphStatus } = useContext(AstroFlowContext);
  const [computeGraph, setComputeGraph] = useState();
  const [nodeId, setNodeId] = useState(null);
  const [nodeLabel, setNodeLabel] = useState(null);

  useEffect(() => {
    fetch(url).then(r => r.json()).then(s => setComputeGraph(s));
  }, [url, computeGraphStatus]);


  // const computeGraph = {
  //   "state": {
  //     "BQ": {
  //       "displayName": "BigQuery Fetcher",
  //       "state": "SKIPPED",
  //       "description": "ai_model_data",
  //       "icon": {
  //         "text": "BQ",
  //         "iconColors": "inverted"
  //       }
  //     },
  //     "Transforms": {
  //       "displayName": "Dataproc Transforms",
  //       "state": "SUCCEEDED",
  //       "description": "/api/fn_clean_data",
  //       "icon": {
  //         "text": "HTTP",
  //         "iconColors": "inverted"
  //       }
  //     },
  //     "Email": {
  //       "displayName": "Email (deleted)",
  //       "description": "Email sent to management",
  //       "icon": {
  //         "name": "web",
  //         "iconset": "shell",
  //         "size": "small"
  //       },
  //       "modifiers": [
  //         "deleted"
  //       ],
  //       "callout": "deleted"
  //     },
  //     "Text": {
  //       "description": "PagerDuty text",
  //       "state": "NO_STATE_RUNTIME",
  //       "icon": {
  //         "name": "topic",
  //         "iconset": "shell",
  //         "size": "small"
  //       }
  //     },
  //     "Text2": {
  //       "description": "PagerDuty text2",
  //       "state": "NO_STATE_RUNTIME",
  //       "icon": {
  //         "name": "topic",
  //         "iconset": "shell",
  //         "size": "small"
  //       }
  //     },
  //     "Broadcast Table": {
  //       "state": "SUCCEEDED",
  //       "description": "Using a custom execution icon",
  //       "subType": "process node",
  //       "icon": {
  //         "name": "performance-dashboard",
  //         "iconset": "shell",
  //         "size": "small"
  //       }
  //     },
  //     "Extract Entities": {
  //       "state": "SKIPPED"
  //     },
  //     "Scan for Importance": {
  //       "state": "SKIPPED"
  //     },
  //     "Send via APIs": {
  //       "state": "PENDING",
  //       "artifactRefs": [
  //         {
  //           "id": "Text Logs",
  //           "path": [
  //             "sub2"
  //           ]
  //         },
  //         {
  //           "id": "Execution Logs",
  //           "path": [
  //             "sub2"
  //           ]
  //         }
  //       ],
  //       "callout": {
  //         "text": "This is a simple callout that goes on for way to long",
  //         "bg": "#1a73e8",
  //         "color": "white"
  //       }
  //     },
  //     "Big Query table": {
  //       "state": "CANCELLED"
  //     },
  //     "AutoML Tables": {
  //       "state": "FAILED"
  //     },
  //     "Output to BigQuery": {
  //       "state": "RUNNING"
  //     },
  //     "CreateFinal": {
  //       "conditionalQuery": "1 == 1",
  //       "artifactRefs": [
  //         {
  //           "id": "BigTable",
  //           "path": [
  //             "sub1"
  //           ]
  //         },
  //         {
  //           "id": "BigTable",
  //           "path": [
  //             "sub1",
  //             "sub1"
  //           ]
  //         }
  //       ]
  //     },
  //     "CustomNode1": {
  //       "templateRef": "outlineBasic",
  //       "width": 480,
  //       "height": 150,
  //       "includeInStepCount": false,
  //       "artifactRefs": [
  //         {
  //           "id": "BigTable",
  //           "path": [
  //             "sub1"
  //           ]
  //         }
  //       ]
  //     },
  //     "TransformedTable": {
  //       "state": "NO_STATE_RUNTIME"
  //     },
  //     "BigTable": {
  //       "state": "NO_STATE_RUNTIME",
  //       "displayName": "Big Table Artifact",
  //       "icon": {
  //         "name": "artifact-model",
  //         "iconset": "cloud_ai",
  //         "size": "large"
  //       }
  //     },
  //     "BigQueryTable": {
  //       "state": "NO_STATE_RUNTIME",
  //       "icon": {
  //         "name": "artifact-datasets",
  //         "iconset": "cloud_ai",
  //         "size": "large",
  //         "color": "#af5cf7"
  //       }
  //     },
  //     "Entities": {
  //       "state": "NO_STATE_RUNTIME",
  //       "description": "People: 512, Places: 82, Indexes: 4012",
  //       "icon": {
  //         "text": "ENT",
  //         "iconColors": "inverted",
  //         "color": "#52cee6"
  //       }
  //     },
  //     "TensorFlow Training": {
  //       "state": "TIMEOUT",
  //       "description": "Defaulted to iteration-2",
  //       "hasControlNode": true,
  //       "treatAsLoop": true,
  //       "selectedLoopId": "it-2",
  //       "icon": {
  //         "name": "group-iterative",
  //         "iconset": "cloud_ai",
  //         "size": "medium"
  //       },
  //       "groupMeta": {
  //         "it-1": {
  //           "state": "SUCCEEDED",
  //           "displayName": "foo",
  //           "groupMeta": {
  //             "Trainer": {
  //               "state": "SUCCEEDED",
  //               "description": "gcr.io/python-images/train-cvvv-v1.1",
  //               "artifactRefs": [
  //                 {
  //                   "id": "BigTable",
  //                   "path": []
  //                 }
  //               ]
  //             },
  //             "Intermediate-Model": {
  //               "state": "NO_STATE_RUNTIME",
  //               "subType": "model",
  //               "icon": {
  //                 "name": "artifact-model",
  //                 "iconset": "cloud_ai",
  //                 "size": "large"
  //               }
  //             },
  //             "Test for accuracy": {
  //               "state": "SUCCEEDED",
  //               "description": "Validates that accuracy is above 85%",
  //               "conditionalQuery": "getModel(\"Intermediate- Model\").accuracy > .85"
  //             }
  //           }
  //         },
  //         "it-2": {
  //           "state": "SUCCEEDED",
  //           "displayName": "bar",
  //           "groupMeta": {
  //             "Trainer": {
  //               "state": "SUCCEEDED",
  //               "description": "gcr.io/python-images/train-cvvv-v1.1",
  //               "artifactRefs": [
  //                 {
  //                   "id": "BigTable",
  //                   "path": []
  //                 }
  //               ]
  //             },
  //             "Intermediate-Model": {
  //               "state": "NO_STATE_RUNTIME",
  //               "subType": "model",
  //               "icon": {
  //                 "name": "artifact-model",
  //                 "iconset": "cloud_ai",
  //                 "size": "large"
  //               }
  //             },
  //             "Test for accuracy": {
  //               "state": "SUCCEEDED",
  //               "description": "Validates that accuracy is above 85%",
  //               "conditionalQuery": "getModel(\"Intermediate- Model\").accuracy > .85"
  //             }
  //           }
  //         },
  //         "it-4": {
  //           "state": "SUCCEEDED",
  //           "displayName": "baz",
  //           "groupMeta": {
  //             "Trainer": {
  //               "state": "SUCCEEDED",
  //               "description": "gcr.io/python-images/train-cvvv-v1.1",
  //               "artifactRefs": [
  //                 {
  //                   "id": "BigTable",
  //                   "path": []
  //                 }
  //               ]
  //             },
  //             "Intermediate-Model": {
  //               "state": "NO_STATE_RUNTIME",
  //               "subType": "model",
  //               "icon": {
  //                 "name": "artifact-model",
  //                 "iconset": "cloud_ai",
  //                 "size": "large"
  //               }
  //             },
  //             "Test for accuracy": {
  //               "state": "SUCCEEDED",
  //               "description": "Validates that accuracy is above 85%",
  //               "conditionalQuery": "getModel(\"Intermediate- Model\").accuracy > .85"
  //             }
  //           }
  //         },
  //         "it-3": {
  //           "displayName": "Iteration 3",
  //           "state": "SKIPPED",
  //           "groupMeta": {
  //             "Trainer": {
  //               "state": "SKIPPED",
  //               "description": "gcr.io/python-images/train-cvvv-v1.1",
  //               "artifactRefs": [
  //                 {
  //                   "id": "BigTable",
  //                   "path": []
  //                 }
  //               ]
  //             },
  //             "Intermediate-Model": {
  //               "state": "NO_STATE_RUNTIME",
  //               "subType": "model",
  //               "icon": {
  //                 "name": "artifact-model",
  //                 "iconset": "cloud_ai",
  //                 "size": "large"
  //               }
  //             },
  //             "Test for accuracy": {
  //               "state": "NOT_TRIGGERED",
  //               "description": "Validates that accuracy is above 85%",
  //               "conditionalQuery": "getModel(\"Intermediate- Model\").accuracy > .85"
  //             }
  //           }
  //         },
  //         "it-5": {
  //           "displayName": "Iteration 5",
  //           "state": "TIMEOUT",
  //           "groupMeta": {
  //             "Trainer": {
  //               "state": "TIMEOUT",
  //               "description": "gcr.io/python-images/train-cvvv-v1.1",
  //               "artifactRefs": [
  //                 {
  //                   "id": "BigTable",
  //                   "path": []
  //                 }
  //               ]
  //             },
  //             "Intermediate-Model": {
  //               "state": "NO_STATE_RUNTIME",
  //               "subType": "model",
  //               "icon": {
  //                 "name": "artifact-model",
  //                 "iconset": "cloud_ai",
  //                 "size": "large"
  //               }
  //             },
  //             "Test for accuracy": {
  //               "state": "PENDING",
  //               "description": "Validates that accuracy is above 85%",
  //               "conditionalQuery": "getModel(\"Intermediate - Model\").accuracy > .85"
  //             }
  //           }
  //         }
  //       }
  //     },
  //     "sub1": {
  //       "state": "CANCELLED",
  //       "hasControlNode": false,
  //       "displayName": "Subdag (sub1)",
  //       "groupMeta": {
  //         "Fake Exec 1": {
  //           "state": "SUCCEEDED",
  //           "conditionalQuery": "len(groupParam) == 1"
  //         },
  //         "BigTable": {
  //           "displayName": "Sub BigTable",
  //           "state": "NO_STATE_RUNTIME",
  //           "icon": {
  //             "name": "artifact-datasets",
  //             "iconset": "cloud_ai",
  //             "size": "large",
  //             "color": "#af5cf7"
  //           }
  //         },
  //         "AutoML Tables": {
  //           "state": "NOT_TRIGGERED",
  //           "conditionalQuery": "len(groupParam) > 2 && thisIsNotTest(ctx)"
  //         },
  //         "Output to BigQuery": {
  //           "state": "CANCELLING"
  //         },
  //         "sub1": {
  //           "state": "NOT_TRIGGERED",
  //           "hasControlNode": true,
  //           "conditionalQuery": "isTaskDone(\"outputToBigQuery\")",
  //           "displayName": "Nested Subdag (sub1)",
  //           "artifactRefs": [
  //             {
  //               "id": "TransformedTable",
  //               "path": []
  //             }
  //           ],
  //           "groupMeta": {
  //             "Fake Exec 1": {
  //               "artifactRefs": [
  //                 {
  //                   "id": "TransformedTable",
  //                   "path": []
  //                 }
  //               ]
  //             },
  //             "BigTable": {
  //               "displayName": "Sub x2 BigTable",
  //               "icon": {
  //                 "name": "artifact-datasets",
  //                 "iconset": "cloud_ai",
  //                 "size": "large",
  //                 "color": "#af5cf7"
  //               }
  //             }
  //           }
  //         },
  //         "CustomNode2": {
  //           "templateRef": "embedYt",
  //           "width": 280,
  //           "height": 37,
  //           "includeInStepCount": false
  //         }
  //       }
  //     },
  //     "sub2": {
  //       "state": "RUNNING",
  //       "hasControlNode": true,
  //       "displayName": "Log Ingestion (sub2)",
  //       "description": "A simple subdag that shows edge-snapping logic above it",
  //       "groupMeta": {
  //         "Node(process, args=1)": {
  //           "displayName": "process",
  //           "description": "1",
  //         },
  //         "Node(process, args=2)": {
  //           "displayName": "process",
  //           "description": "2",
  //         },
  //         "Text Logs": {
  //           "state": "NO_STATE_RUNTIME",
  //           "icon": {
  //             "name": "logs",
  //             "iconset": "shell",
  //             "size": "small"
  //           }
  //         },
  //         "Execution Logs": {
  //           "state": "NO_STATE_RUNTIME",
  //           "icon": {
  //             "name": "logs",
  //             "iconset": "shell",
  //             "size": "small"
  //           }
  //         },
  //         "Scan Texts": {
  //           "state": "SUCCEEDED",
  //           "artifactRefs": [
  //             {
  //               "id": "Text",
  //               "path": []
  //             },
  //             {
  //               "id": "Text2",
  //               "path": []
  //             }
  //           ]
  //         },
  //         "Label Execution Logs": {
  //           "state": "RUNNING",
  //           "artifactRefs": [
  //             {
  //               "id": "Text",
  //               "path": []
  //             },
  //             {
  //               "id": "Entities",
  //               "path": []
  //             }
  //           ]
  //         }
  //       }
  //     }
  //   },
  //   "skeleton": [
  //     {
  //       "id": "BQ",
  //       "type": "execution",
  //       "next": [
  //         {
  //           "id": "Transforms",
  //           "type": "execution",
  //           "next": [
  //             {
  //               "id": "TransformedTable",
  //               "type": "artifact",
  //               "next": [
  //                 {
  //                   "id": "Big Query table",
  //                   "type": "execution",
  //                   "next": [
  //                     {
  //                       "id": "BigTable",
  //                       "type": "artifact",
  //                       "next": [
  //                         {
  //                           "edgeLabel": "storage (red-edge)",
  //                           "edgeOpts": {
  //                             "toMarkerStyle": "arrow",
  //                             "color": "#E94235",
  //                             "weight": 3
  //                           },
  //                           "id": "AutoML Tables",
  //                           "type": "execution",
  //                           "next": [
  //                             {
  //                               "edgeOpts": {
  //                                 "color": "#E94235",
  //                                 "toMarkerStyle": "arrow"
  //                               },
  //                               "id": "Output to BigQuery",
  //                               "type": "execution",
  //                               "next": [
  //                                 {
  //                                   "edgeOpts": {
  //                                     "toMarkerStyle": "arrow"
  //                                   },
  //                                   "id": "BigQueryTable",
  //                                   "type": "artifact",
  //                                   "next": [
  //                                     {
  //                                       "edgeOpts": {
  //                                         "toMarkerStyle": "arrow"
  //                                       },
  //                                       "id": "CreateFinal",
  //                                       "type": "execution"
  //                                     }
  //                                   ]
  //                                 }
  //                               ]
  //                             }
  //                           ]
  //                         },
  //                         {
  //                           "edgeLabel": "input-data",
  //                           "id": "TensorFlow Training",
  //                           "type": "group",
  //                           "definition": [
  //                             {
  //                               "id": "it-1",
  //                               "type": "group",
  //                               "definition": [
  //                                 {
  //                                   "id": "Trainer",
  //                                   "type": "execution",
  //                                   "next": [
  //                                     {
  //                                       "id": "Intermediate-Model",
  //                                       "type": "artifact",
  //                                       "next": [
  //                                         {
  //                                           "id": "Test for accuracy",
  //                                           "type": "execution"
  //                                         }
  //                                       ]
  //                                     }
  //                                   ]
  //                                 }
  //                               ]
  //                             },
  //                             {
  //                               "id": "it-2",
  //                               "type": "group",
  //                               "definition": [
  //                                 {
  //                                   "id": "Trainer",
  //                                   "type": "execution",
  //                                   "next": [
  //                                     {
  //                                       "id": "Intermediate-Model",
  //                                       "type": "artifact",
  //                                       "next": [
  //                                         {
  //                                           "id": "Test for accuracy",
  //                                           "type": "execution"
  //                                         }
  //                                       ]
  //                                     }
  //                                   ]
  //                                 }
  //                               ]
  //                             },
  //                             {
  //                               "id": "it-3",
  //                               "type": "group",
  //                               "definition": [
  //                                 {
  //                                   "id": "Trainer",
  //                                   "type": "execution",
  //                                   "next": [
  //                                     {
  //                                       "id": "Intermediate-Model",
  //                                       "type": "artifact",
  //                                       "next": [
  //                                         {
  //                                           "id": "Test for accuracy",
  //                                           "type": "execution"
  //                                         }
  //                                       ]
  //                                     }
  //                                   ]
  //                                 }
  //                               ]
  //                             },
  //                             {
  //                               "id": "it-4",
  //                               "type": "group",
  //                               "definition": [
  //                                 {
  //                                   "id": "Trainer",
  //                                   "type": "execution",
  //                                   "next": [
  //                                     {
  //                                       "id": "Intermediate-Model",
  //                                       "type": "artifact",
  //                                       "next": [
  //                                         {
  //                                           "id": "Test for accuracy",
  //                                           "type": "execution"
  //                                         }
  //                                       ]
  //                                     }
  //                                   ]
  //                                 }
  //                               ]
  //                             },
  //                             {
  //                               "id": "it-5",
  //                               "type": "group",
  //                               "definition": [
  //                                 {
  //                                   "id": "Trainer",
  //                                   "type": "execution",
  //                                   "next": [
  //                                     {
  //                                       "id": "Intermediate-Model",
  //                                       "type": "artifact",
  //                                       "next": [
  //                                         {
  //                                           "id": "Test for accuracy",
  //                                           "type": "execution"
  //                                         }
  //                                       ]
  //                                     }
  //                                   ]
  //                                 }
  //                               ]
  //                             }
  //                           ],
  //                           "next": [
  //                             {
  //                               "id": "Output to BigQuery",
  //                               "type": "execution"
  //                             },
  //                             {
  //                               "edgeLabel": "SUB-DAG 1",
  //                               "id": "sub1",
  //                               "type": "group",
  //                               "definition": [
  //                                 {
  //                                   "id": "Output to BigQuery",
  //                                   "type": "execution"
  //                                 },
  //                                 {
  //                                   "id": "Fake Exec 1",
  //                                   "type": "execution",
  //                                   "next": [
  //                                     {
  //                                       "id": "BigTable",
  //                                       "type": "artifact",
  //                                       "next": [
  //                                         {
  //                                           "edgeLabel": "storage",
  //                                           "id": "AutoML Tables",
  //                                           "type": "execution"
  //                                         },
  //                                         {
  //                                           "edgeLabel": "SUB-DAG Nested 1",
  //                                           "id": "sub1",
  //                                           "type": "group",
  //                                           "definition": [
  //                                             {
  //                                               "id": "Output to BigQuery",
  //                                               "type": "execution"
  //                                             },
  //                                             {
  //                                               "id": "Fake Exec 1",
  //                                               "type": "execution",
  //                                               "next": [
  //                                                 {
  //                                                   "edgeOpts": {
  //                                                     "weight": 10
  //                                                   },
  //                                                   "id": "BigTable",
  //                                                   "type": "artifact",
  //                                                   "next": [
  //                                                     {
  //                                                       "edgeLabel": "storage",
  //                                                       "id": "AutoML Tables",
  //                                                       "type": "execution"
  //                                                     }
  //                                                   ]
  //                                                 }
  //                                               ]
  //                                             }
  //                                           ],
  //                                           "next": [
  //                                             {
  //                                               "edgeLabel": "custom-video",
  //                                               "id": "CustomNode2",
  //                                               "type": "artifact"
  //                                             }
  //                                           ]
  //                                         }
  //                                       ]
  //                                     }
  //                                   ]
  //                                 }
  //                               ],
  //                               "next": [
  //                                 {
  //                                   "id": "CreateFinal",
  //                                   "type": "execution"
  //                                 },
  //                                 {
  //                                   "id": "CustomNode1",
  //                                   "type": "execution"
  //                                 }
  //                               ]
  //                             }
  //                           ]
  //                         }
  //                       ]
  //                     }
  //                   ]
  //                 },
  //                 {
  //                   "edgeOpts": {
  //                     "toMarkerStyle": "arrow"
  //                   },
  //                   "id": "sub1",
  //                   "type": "group"
  //                 },
  //                 {
  //                   "id": "Broadcast Table",
  //                   "type": "execution",
  //                   "next": [
  //                     {
  //                       "id": "Email",
  //                       "type": "artifact",
  //                       "next": [
  //                         {
  //                           "id": "Send via APIs",
  //                           "type": "execution"
  //                         }
  //                       ]
  //                     },
  //                     {
  //                       "id": "Text2",
  //                       "type": "artifact",
  //                       "next": [
  //                         {
  //                           "edgeLabel": "obtuse-edge",
  //                           "id": "sub2",
  //                           "type": "group"
  //                         }
  //                       ]
  //                     },
  //                     {
  //                       "id": "Text",
  //                       "type": "artifact",
  //                       "next": [
  //                         {
  //                           "edgeLabel": "SUB-DAG 2",
  //                           "id": "sub2",
  //                           "type": "group",
  //                           "definition": [
  //                             {
  //                               "id": "Node(process, args=1)",
  //                               "type": "execution",
  //                               "next": [
  //                                 {
  //                                   "id": "gather",
  //                                   "type": "execution",
  //                                 }
  //                               ]
  //                             },
  //                             {
  //                               "id": "Node(process, args=2)",
  //                               "type": "execution",
  //                               "next": [
  //                                 {
  //                                   "id": "gather",
  //                                   "type": "execution",
  //                                 }
  //                               ]
  //                             }
  //                             // {
  //                             //   "id": "Scan Texts",
  //                             //   "type": "execution",
  //                             //   "next": [
  //                             //     {
  //                             //       "id": "Text Logs",
  //                             //       "type": "artifact"
  //                             //     }
  //                             //   ]
  //                             // },
  //                             // {
  //                             //   "id": "Label Execution Logs",
  //                             //   "type": "execution",
  //                             //   "next": [
  //                             //     {
  //                             //       "id": "Execution Logs",
  //                             //       "type": "artifact"
  //                             //     }
  //                             //   ]
  //                             // }
  //                           ],
  //                           "next": [
  //                             {
  //                               "id": "Send via APIs",
  //                               "type": "execution"
  //                             }
  //                           ]
  //                         }
  //                       ]
  //                     },
  //                     {
  //                       "id": "Extract Entities",
  //                       "type": "execution",
  //                       "next": [
  //                         {
  //                           "id": "Scan for Importance",
  //                           "type": "execution",
  //                           "next": [
  //                             {
  //                               "id": "sub2",
  //                               "type": "group"
  //                             }
  //                           ]
  //                         },
  //                         {
  //                           "id": "Entities",
  //                           "type": "artifact",
  //                           "next": [
  //                             {
  //                               "id": "sub2",
  //                               "type": "group"
  //                             }
  //                           ]
  //                         }
  //                       ]
  //                     }
  //                   ]
  //                 }
  //               ]
  //             }
  //           ]
  //         }
  //       ]
  //     }
  //   ]
  // }

  // const { computeGraph } = useContext(AstroFlowContext);

  // const computeGraph = {
  //   'skeleton': [
  //     {
  //       "edgeLabel": "SUB-DAG 2",
  //       "id": "sub2",
  //       "type": "group",
  //       "definition": [{
  //         "id": "Scan Texts",
  //         "type": "execution",
  //         "next": [{
  //           "id": "Text Logs",
  //           "type": "artifact"
  //         }]
  //       }, {
  //         "id": "Label Execution Logs",
  //         "type": "execution",
  //         "next": [{
  //           "id": "Execution Logs",
  //           "type": "artifact",
  //         }]
  //       }],
  //       "next": [{
  //         "id": "Send via APIs",
  //         "type": "execution"
  //       }]
  //     },
  //     {
  //       'id': '3117593921912447883',
  //       'type': 'execution',
  //       "next": [{
  //         "id": "sub2",
  //         "type": "group"
  //       }]
  //     },
  //     // This is the little input artifact thing
  //     {
  //       "id":
  //         "Text2",
  //       "type":
  //         "artifact",
  //       "next": [{
  //         // "edgeLabel": "obtuse-edge",
  //         "id": "sub2",
  //         "type": "group"
  //       }]
  //     },
  //   ],
  //   'state': {
  //     '3117593921912447883': {
  //       'displayName': 'load_metadata',
  //       'description': 'load_metadata',
  //       'icon': {
  //         // 'text': 'SCIM',
  //         'iconColors': 'inverted'
  //       }
  //     },
  //     "Text2": {
  //       "icon": "metrics",
  //     },
  //     "sub2": {
  //       "state":
  //         "RUNNING",
  //       "hasControlNode":
  //         true,
  //       "displayName":
  //         "Log Ingestion (sub2)",
  //       "description":
  //         "A simple subdag that shows edge-snapping logic above it",
  //       "groupMeta": {
  //         "Text Logs": {
  //           "state": "NO_STATE_RUNTIME",
  //           "icon": {
  //             "name": "logs",
  //             "iconset": "shell",
  //             "size": "small"
  //           }
  //         },
  //         "Execution Logs": {
  //           "state": "NO_STATE_RUNTIME",
  //           "icon": {
  //             "name": "logs",
  //             "iconset": "shell",
  //             "size": "small"
  //           }
  //         },
  //         "Scan Texts": {
  //           "state":
  //             "SUCCEEDED",
  //           "artifactRefs": [{
  //             "id": "Text",
  //             "path": []
  //           }, {
  //             "id": "Text2",
  //             "path": []
  //           }]
  //         },
  //         "Label Execution Logs": {
  //           "state":
  //             "RUNNING",
  //           "artifactRefs": [{
  //             "id": "Text",
  //             "path": []
  //           }, {
  //             "id": "Entities",
  //             "path": []
  //           }]
  //         }
  //       }
  //     }
  //   }
  // }

  const graphRef = React.useRef();

  useEffect(() => {
    // TODO: Better typing for the ref
    const graphComponent = graphRef.current;

    if (!graphComponent) {
      return;
    }

    const onSelectedNodeChange = async (event) => {
      console.log(event);
      // console.log(event.detail.node.options.imageUrl);
      // const url = event.detail?.node?.options?.imageUrl;
      // const res = await fetch(url, { method: 'HEAD' });

      const node = event?.detail?.node ?? null;
      const id =
        node?.options?.nodeId ??
        node?.id ??
        node?.name ??
        node?.label ??
        null;

      setNodeId(id);
      setNodeLabel(node?.label ?? node?.name ?? null);
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
  }, [computeGraph])

  React.useEffect(() => {
    document.body.classList.add('workflow-graph-theme-dark');
    return () => document.body.classList.remove('workflow-graph-theme-dark');
  }, []);

  if (!computeGraph) {
    return;
  }

  return <>
    <div className="WorkflowGraph">
      <div className="example-wrapper">
        <google-workflow-graph ref={graphRef}></google-workflow-graph>
        <workflow-graph-color-theme-loader />
      </div>
    </div>
    <NodeStatusDialog
      open={Boolean(nodeId)}
      nodeId={nodeId}
      nodeLabel={nodeLabel}
      onClose={() => setNodeId(null)}
    />
  </>;
}
