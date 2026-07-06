require("dotenv").config();
const express = require("express");
const cors = require("cors");
const connectDB = require("./src/config/db");
const studentRoutes = require("./src/routes/studentRoutes");

const app = express();
app.use(cors());
app.use(express.json());

connectDB();

app.use("/api/students", studentRoutes);

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Server running on ${PORT}`));
