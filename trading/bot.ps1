$ProjectFolder = Split-Path -Parent $MyInvocation.MyCommand.Path

$PythonExe = "$ProjectFolder\venv\Scripts\python.exe"

# Open Positions (09:00 weekdays)
$ActionOpen = New-ScheduledTaskAction -Execute $PythonExe -Argument "trading/execute.py open" -WorkingDirectory $ProjectFolder
$TriggerOpen = New-ScheduledTaskTrigger -Weekly -At "09:00AM" -DaysOfWeek Mon,Tue,Wed,Thu,Fri
Register-ScheduledTask -TaskName "Open Positions" -Action $ActionOpen -Trigger $TriggerOpen -Description "Bot: Open positions at market open" -Force

# Close Positions (16:45 weekdays)
$ActionClose = New-ScheduledTaskAction -Execute $PythonExe -Argument "trading/execute.py close" -WorkingDirectory $ProjectFolder
$TriggerClose = New-ScheduledTaskTrigger -Weekly -At "04:45PM" -DaysOfWeek Mon,Tue,Wed,Thu,Fri
Register-ScheduledTask -TaskName "Close Positions" -Action $ActionClose -Trigger $TriggerClose -Description "Bot: Close positions before market close" -Force

Write-Host "Tasks created successfully."