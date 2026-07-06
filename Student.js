import mongoose from "mongoose";

const studentSchema = new mongoose.Schema({
  name: String,
  email: String,
  marks: Number,
  attendance: Number,
  risk_score: Number,
  risk_level: String,
  last_notified: Date
});

// Force correct collection name
export default mongoose.model("Student", studentSchema, "students");