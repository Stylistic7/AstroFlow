import React from "react";
import { Button, Icon, InputGroup, Navbar, NavbarGroup, NavbarHeading, NavbarDivider, Tag } from "@blueprintjs/core";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";

const AstroFlowNavBar = () => {
  return (
    <Navbar className="bp5-dark" style={{
      position: "fixed",
      top: 0,
      left: 0,
      width: "100%",
      zIndex: 1000, // Ensures it stays above other content
    }}>
      <NavbarGroup align="left">
        <Button icon="menu" minimal />
        <img src="/static/images/astroflow_logo_white.png" style={{ height: 30 }} />
      </NavbarGroup>

      <NavbarGroup align="center">
        {/* Search Bar */}
        <InputGroup leftIcon="search" placeholder="Search..." />
      </NavbarGroup>

      <NavbarGroup align="right">
        {/* Notification Icons */}
        <Button minimal icon="notifications">
          {/* Badge */}
          <Tag round intent="danger" style={{ position: "absolute", top: "5px", right: "5px", fontSize: "10px" }}>3</Tag>
        </Button>

        <Button minimal icon="envelope">
          <Tag round intent="primary" style={{ position: "absolute", top: "5px", right: "5px", fontSize: "10px" }}>5</Tag>
        </Button>

        {/* User Profile Icon */}
        <Button minimal icon="user" />
      </NavbarGroup>
    </Navbar>
  );
};

export default AstroFlowNavBar;
