import React, { useEffect, useState } from "react";
import EnhancedTable from "@jeremyling/react-material-ui-enhanced-table";
import { amber, green, indigo, lightGreen, red } from "@mui/material/colors";
import { Block, MoveToInbox } from "@mui/icons-material";
import TargetTable from "./TargetTable.jsx";
import './MosaicTable.css'; 

// TODO: Unify this.
function style_for_filter(filter) {
 const color = {
    'L': [51, 51, 51],
    'R': [230, 0, 0],
    'G': [0, 204, 0],
    'B': [0, 0, 204],
    'H': [159, 43, 104],
    'S': [140, 0, 0],
    'O': [0, 204, 204],
  }[filter];
  
  return {
    border: `1px solid rgba(${color[0]}, ${color[1]}, ${color[2]}, 1.0)`,
    margin: `2px`,
  };
}

export default function Mosaics(props) {
  const [mosaics, setMosaics] = useState([]);
  const [targets, setTargets] = useState([]);
  // const [stacks, setStacks] = useState([]);
  const [totalCount, setTotalCount] = useState();
  const [tableLoading, setTableLoading] = useState(false);
  const [order, setOrder] = useState("");
  const [orderBy, setOrderBy] = useState("");
  const [page, setPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const [selectedRow, setSelectedRow] = useState();
  const [selectedItems, setSelectedItems] = useState({});

  const [openRows, setOpenRows] = useState({});

  const [filters, setFilters] = useState();

  const [notificationsCount, setNotificationsCount] = useState();

  const headers = [
    {
      key: "collapse",
      collapse: true,
    },
    { key: "name", attribute: "name", label: "Mosaic" },
    // {
    //   key: "image",
    //   label: "Thumbnails",
    //   component: (target) => {
    //     let output = [];
    //     if (stacks) {
    //       for (let id of target.stacks) {
    //         console.log('looking up stack id ' + id);
    //         let stack = stacks.find(function (element) {
    //           return element.id == id;
    //         });
    //         console.log('stack ' + stack);

    //         if (stack) {
    //           // TODO: Assert stack
    //           // TODO: Sort images by filter
    //           let stack_thumbnail_image_url = `/stack_thumbnail_image/${stack.target}/${stack.rig}/${stack.filter}`;
    //           output.push(
    //             <img src={stack_thumbnail_image_url}
    //                  width="100px"
    //                  style={style_for_filter(stack.filter)} />);
    //         }
    //       }
    //     }
    //     return output;
    //   },
    //   stopPropagation: true,
    // },
    // {
    //   key: "status",
    //   attribute: "status",
    //   label: "Status",
    //   chip: true,
    //   chipColor: {
    //     "In Progress": [amber[50], amber[200]],
    //     Completed: [green[50], green[200]],
    //   },
    // },
    // {
    //   attribute: "updated_at",
    //   datetime: true,
    //   label: "Updated",
    // },
    // {
    //   multiField: true,
    //   key: "customer",
    //   multiFieldData: [
    //     {
    //       attribute: "user.name",
    //     },
    //     {
    //       attribute: "user.email",
    //     },
    //     {
    //       attribute: "user.phone",
    //     },
    //   ],
    //   html: `
    //     <div><strong>{{0}}</strong></div>
    //     <div style="color: #777">{{1}}</div>
    //     <div style="color: #777">{{2}}</div>
    //   `,
    //   label: "Customer",
    // },
    // {
    //   attribute: "total",
    //   label: "Total",
    //   numeric: true,
    //   price: true,
    // },
    {
      key: "actions",
      actions: [
        {
          id: "process",
          icon: <MoveToInbox />,
          tooltip: "Process",
          onClick: (event, row) => handleAction(event, "process", row),
          color: amber[400],
          hideCondition: (row) => {},
        },
        {
          id: "cancel",
          icon: <Block />,
          tooltip: "Cancel",
          onClick: (event, row) => handleAction(event, "cancel", row),
          color: indigo[400],
        },
      ],
      label: "Actions",
    },
  ];

  const collapseContent = () => {
    return mosaics.map((mosaic) => {
      let polygons = [];
      for (let target_id of mosaic.target_ids) {
        let target = targets[target_id];
        if (!target) continue;

        let points_string = target.footprint.join(' ');
        console.log(points_string);

        polygons.push(
          <polygon className='mosaic_polygon' points={points_string} />
        );
      }

      let viewbox = mosaic.viewbox.join(' ');

      return (
        <div>
          <svg viewBox={viewbox} style={{display: "block", width: "500px", margin: "auto"}} xmlns="http://www.w3.org/2000/svg">
            {polygons}
          </svg>
          <TargetTable key="{mosaic.id}" target_ids={mosaic.target_ids} />
        </div>
      );
    });
  };

  const handleRowClick = (event, item) => {
    // setSelectedRow(item);
  };

  function handleCollapsibleTableRowClick(event, row) {}

  function getSelectibleItems(row) {}

  function stringifyEvent(e) {
    const obj = {};
    for (let k in e) {
      obj[k] = e[k];
    }
    return JSON.stringify(obj, (k, v) => {
      if (v instanceof Node) return 'Node';
      if (v instanceof Window) return 'Window';
      return v;
    }, ' ');
  }

  function updateOpenRows(event, key, value) {
    console.log('updateOpenRows event ' + stringifyEvent(event) + ' key ' + key + ' value ' + value);
    event.stopPropagation();

    let updated = JSON.parse(JSON.stringify(openRows));
    console.log('openRows ' + JSON.stringify(openRows));
    updated[key] = value;
    setOpenRows(updated);
  }

  const handleRequestSort = (event, property) => {
    if (orderBy !== property) {
      setOrder("asc");
      setOrderBy(property);
      return;
    }

    switch (order) {
      case "asc":
        setOrder("desc");
        break;
      case "desc":
        setOrder("");
        setOrderBy("");
        break;
      default:
        break;
    }
  };

  const handlePageChange = (event, newPage) => {
    setPage(newPage);
  };

  const handleRowsPerPageChange = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(1);
  };

  const handleDateChange = (type, value) => {
    // Update filters
  };

  function handleAction(event, key, row = null) {}

  useEffect(() => {
    console.log('fetching targets...');

    // TODO: Don't download these multiple times, because they're pretty big.
    // TODO: Remove the FileLists from the objects before sending them across
    // the network.
    fetch('/mosaics').then(res => res.json()).then(mosaics => {
      setMosaics(Object.values(mosaics));  
    });
    fetch('/targets').then(res => res.json()).then(targets => {
      setTargets(Object.values(targets));  
    });
  }, [order, orderBy, rowsPerPage, page, filters]);

  return (
    <EnhancedTable
      rows={mosaics || []}
      totalCount={totalCount}
      headers={headers}
      order={order}
      orderBy={orderBy}
      loading={tableLoading}
      page={page}
      rowsPerPage={rowsPerPage}
      handleRowClick={handleRowClick}
      handleRequestSort={handleRequestSort}
      handlePageChange={handlePageChange}
      handleRowsPerPageChange={handleRowsPerPageChange}
      handleActionClick={(event, key) => handleAction(event, key)}
      handleDateChange={handleDateChange}
      // dates={{ from: filters.dateFrom, to: filters.dateTo }}
      actionButtons={["dateFilters", "refresh"]}
      collapsible={true}
      collapseContent={collapseContent()}
      refreshBadgeCount={notificationsCount}
      disableSelection={true}
      openRows={openRows}
      handleCollapseIconClick={updateOpenRows}
      identifier="id"
    />
  );
}
