import React, { useEffect, useState } from "react";
import EnhancedTable from "@jeremyling/react-material-ui-enhanced-table";
import { amber, green, indigo, lightGreen, red } from "@mui/material/colors";
import { Block, MoveToInbox } from "@mui/icons-material";
import StackTable from "./StackTable.jsx";

export default function Targets(props) {
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
      key: "collapse",
      collapse: true,
    },
    { key: "name", attribute: "name", label: "Target" },
    // {
    //   key: "image",
    //   label: "Thumbnail",
    //   component: (data) => (
    //     <img src={data.thumbnail_image_url} alt={data.name} width="200px" loading="lazy" />
    //   ),
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
    if (data) {
      return data.map((target) => (
        <StackTable stacks={target.stacks} />
      ));
    }
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
    console.log('fetching targets...');
    fetch('/targets').then(res => res.json()).then(data => {
      setData(data);
    });
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
