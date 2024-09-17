import Icon from "@mui/material/Icon";

// Pages
import ContactUs from "pages/LandingPages/ContactUs";

const routes = [
  {
    name: "pages",
    icon: <Icon>dashboard</Icon>,
    columns: 1,
    rowsPerColumn: 2,
    collapse: [
      {
        name: "home",
        collapse: [
          {
            name: "home",
            route: "/home",
            component: <ContactUs />,
          },
        ],
      },
    ],
  },
];

export default routes;
