import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Checkbox, FormGroup, FormControlLabel } from '@mui/material';

export default function AutoStackCheckbox({ object_type, object_id }) {
  const navigate = useNavigate();

  const [shopping_cart, set_shopping_cart] = useState([]);

  // TODO: Move this up the component hierarchy.
  useEffect(() => {
    fetch(`/get_shopping_cart`)
      .then(r => r.json())
      .then(j => set_shopping_cart(j))
  }, []);

  const autoStackButtonChanged = (event) => {
    console.log('autoStackButtonChanged', event.target.checked);
    if (event.target.checked) {
      fetch(`/add_to_shopping_cart/${object_type}/${object_id}`)
        .then(r => r.json())
        .then(j => set_shopping_cart(j));
    } else {
      fetch(`/remove_from_shopping_cart/${object_type}/${object_id}`)
        .then(r => r.json())
        .then(j => set_shopping_cart(j));;
    }
  };

  // console.log('shopping_cart', shopping_cart);

  const in_shopping_cart = () => {
    for (const [t, i] of shopping_cart) {
      console.log('t', t, 'i', i);
      if (t == object_type && i == object_id) {
        return true;
      }
    }

    return false;
  };

  return (
    <FormGroup>
      <FormControlLabel control={
        <Checkbox
          checked={in_shopping_cart()}
          onChange={autoStackButtonChanged}
          size="small" />
      } label="Auto-Stack" />
    </FormGroup>
  );
}