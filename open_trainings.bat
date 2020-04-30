set mypath=%cd%
%mypath%
for /l %%x in (1, 1, 100) do (
   timeout 5
   start opera.exe "%mypath%\Training Predictions (%%x).html"
)