# PowerShell script to create two daily tasks: open and close positions

# Get the folder where this script resides (your project folder)
$ProjectFolder = Split-Path -Parent $MyInvocation.MyCommand.Path

# Path to Python (assumes Python is in PATH)
$PythonExe = "python.exe"

# Open Positions Task (09:00)
$ActionOpen = New-ScheduledTaskAction -Execute $PythonExe -Argument "trading/execute.py open" -WorkingDirectory $ProjectFolder
$TriggerOpen = New-ScheduledTaskTrigger -Daily -At "09:00AM" -DaysOfWeek Mon,Tue,Wed,Thu,Fri
Register-ScheduledTask -TaskName "Open Positions" -Action $ActionOpen -Trigger $TriggerOpen -Description "Bot: Open positions at market open" -Force

# Close Positions Task (16:45)
$ActionClose = New-ScheduledTaskAction -Execute $PythonExe -Argument "trading/execute.py close" -WorkingDirectory $ProjectFolder
$TriggerClose = New-ScheduledTaskTrigger -Daily -At "04:45PM" -DaysOfWeek Mon,Tue,Wed,Thu,Fri
Register-ScheduledTask -TaskName "Close Positions" -Action $ActionClose -Trigger $TriggerClose -Description "Bot: Close positions before market close" -Force

Write-Host "Tasks created successfully."