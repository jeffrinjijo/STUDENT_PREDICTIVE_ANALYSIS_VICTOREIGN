import dotenv from "dotenv";
dotenv.config();

import mongoose from "mongoose";
import sgMail from "@sendgrid/mail";
import Student from "./models/Student.js";

console.log("MONGO URI =", process.env.MONGO_URI);

sgMail.setApiKey(process.env.SENDGRID_API_KEY);

async function notify() {
  try {
    // Ensure correct DB
    await mongoose.connect(process.env.MONGO_URI, {
      dbName: "notification"
    });

    console.log("MongoDB Connected for notification");

    // Debug: show all students
    const all = await Student.find({});
    console.log("Total students in DB:", all.length);
    console.log(all);

    const THRESHOLD = 0.6;

    // ✅ FIXED QUERY (works 100%)
    const students = await Student.find({
      risk_score: { $gte: THRESHOLD }
    });

    console.log(`High-risk students found: ${students.length}`);

    for (let s of students) {
      const msg = {
        to: s.email,
        from: process.env.FROM_EMAIL,
        subject: "⚠️ Academic Warning — Performance At Risk",
        html: `
          <p>Dear ${s.name},</p>
          <p>Your academic performance indicates a <strong>HIGH RISK</strong> status.</p>

          <p><b>Marks:</b> ${s.marks}%<br>
          <b>Attendance:</b> ${s.attendance}%<br>
          <b>Risk Score:</b> ${(s.risk_score * 100).toFixed(1)}%</p>

          <p>Please take immediate action:</p>
          <ul>
            <li>Attend all upcoming classes</li>
            <li>Submit pending assignments</li>
            <li>Meet your faculty mentor</li>
          </ul>

          <p>Regards,<br>Academic Monitoring System</p>
        `
      };

      try {
        await sgMail.send(msg);
        console.log(`Email sent to ${s.email}`);

        // Update last notified
        s.last_notified = new Date();
        await s.save();
      } catch (err) {
        console.log("Error sending email:", err.message);
      }
    }

    process.exit();
  } catch (err) {
    console.error("Unexpected Error:", err);
  }
}

notify();