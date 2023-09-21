import React, { useEffect, useState } from "react";
import EnhancedTable from "@jeremyling/react-material-ui-enhanced-table";
import { amber, green, indigo, lightGreen, red } from "@mui/material/colors";
import { Block, MoveToInbox } from "@mui/icons-material";

function exposure_time_string(exposure_time_s) {
  let hours = Math.floor(exposure_time_s / 3600)
  let minutes = Math.floor(exposure_time_s % 3600 / 60)
  return hours + ' h ' + minutes.toString().padStart(2, '0') + ' m';
}

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
  };
}

export default function Stacks(props) {
  const [data, setData] = useState([]);
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
      key: "checkbox",
      checkbox: true,
    },
    // {
    //   key: "image",
    //   label: "Image",
    //   component: (data) => (
    //     <img src={data.image.url} alt={data.name} width="200px" loading="lazy" />
    //   ),
    //   stopPropagation: true,
    // },
    // {
    //   key: "plot",
    //   label: "Plot",
    //   component: (data) => (
    //     <iframe src={data.plot.url} width="300" height="200"></iframe>
    //   ),
    //   stopPropagation: true,
    // },
    {
      multiField: true,
      key: "stack",
      multiFieldData: [
        {
          attribute: "rig",
        },
        {
          attribute: "filter",
        },
        {
          attribute: "exposure_time_s",
        },
      ],
      component: (data) => (
        <div>
          <div><strong>Rig: {data.rig}, Filter: {data.filter}</strong></div>
          <div><strong style={{color: '#777'}}>
            Exposure: {exposure_time_string(data.exposure_time_s)}
          </strong></div>
        </div>
      ),
      label: "Stack",
      minWidth: "200px",
    },
    {
      key: "video",
      label: "Raw Blink Video",
      component: (data) => (
        <video width="300" loop autoPlay muted style={style_for_filter(data.filter)}>
          <source src={`/stack_thumbnail_video/${data.target}/${data.rig}/${data.filter}`} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      ),
      stopPropagation: true,
    },
    {
      key: "image",
      label: "Flat-Corrected",
      component: (data) => (
        <img src={`/stack_thumbnail_image/${data.target}/${data.rig}/${data.filter}`} alt={data.name} width="200px" loading="lazy" />
      ),
      stopPropagation: true,
    },
    // {
    //   key: "status",
    //   attribute: "status",
    //   label: "Status",
    //   chip: true,
    //   chipColor: {
    //     Pending: [amber[50], amber[200]],
    //     Paid: [lightGreen[50], lightGreen[200]],
    //     Processed: [indigo[50], indigo[200]],
    //     Completed: [green[50], green[200]],
    //   },
    // },
    // {
    //   attribute: "updated_at",
    //   datetime: true,
    //   label: "Updated",
    // },
    // { attribute: "quantity", label: "Qty", numeric: true },
    {
      key: "actions",
      actions: [
        {
          id: "process",
          icon: <MoveToInbox />,
          tooltip: "Process",
          onClick: (event, row) => handleAction(event, "process", row),
          color: amber[400],
          hideCondition: (data) => {},
        },
        {
          id: "cancel-target",
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
  };

  const handleRowClick = (event, item) => {
    setSelectedRow(item);
  };

  function handleCollapsibleTableRowClick(event, row) {}

  function getSelectibleItems(row) {}

  function updateOpenRows(event, key, value) {
    event.stopPropagation();

    let updated = JSON.parse(JSON.stringify(openRows));
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
    Promise.all(
      props.stack_ids.map(id => fetch('/stack/' + id).then(res => res.json()))
    ).then(data => setData(data));    
  }, [order, orderBy, rowsPerPage, page, filters]);

  return (
    <EnhancedTable
      rows={data || []}
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
    />
  );
}
