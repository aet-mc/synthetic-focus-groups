/**
 * Google Apps Script — deploy as Web App to collect survey responses.
 * 
 * Setup:
 * 1. Go to https://script.google.com
 * 2. Create new project, paste this code
 * 3. Deploy → New Deployment → Web App
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 4. Copy the URL and paste into index.html
 * 
 * Responses are stored in a Google Sheet (auto-created on first submission).
 */

const SHEET_NAME = "SFG Validation Responses";

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    
    // Get or create spreadsheet
    let ss;
    const files = DriveApp.getFilesByName(SHEET_NAME);
    if (files.hasNext()) {
      ss = SpreadsheetApp.open(files.next());
    } else {
      ss = SpreadsheetApp.create(SHEET_NAME);
    }
    
    const sheet = ss.getActiveSheet();
    
    // Add headers on first row if empty
    if (sheet.getLastRow() === 0) {
      const headers = Object.keys(data);
      sheet.appendRow(headers);
    }
    
    // Add data row
    const headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    const row = headers.map(h => data[h] || "");
    sheet.appendRow(row);
    
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true, row: sheet.getLastRow() }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  return ContentService
    .createTextOutput(JSON.stringify({ status: "Survey collection endpoint active" }))
    .setMimeType(ContentService.MimeType.JSON);
}
