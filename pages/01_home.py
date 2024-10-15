import streamlit as st 
from streamlit_calendar import calendar
import datetime
import time


col1, col2, col3, = st.columns(3, gap="large")
time = time.strftime("%H:%M:%S", time.localtime())
current_time = datetime.datetime.now()
c_time = current_time.second
col1.title("Welcome!")
col3.metric(label='Time', value=time, delta=c_time)

st.header('This is a _ROCK_ _FRAGMENTATION_ :red[WEB APP]')

st.markdown(':blue[**The calendar below is for scheduling drilling and blasting rounds sequences in different zones of the mine**.]')
with st.container(border=True, height=500):
    mode = st.selectbox(
    "Calendar Mode:",
    (
        "daygrid",
        "timegrid",
        "timeline",
        "resource-daygrid",
        "resource-timegrid",
        "resource-timeline",
        "list",
        "multimonth",
    ),
)
    calendar_options = {
        "editable": "true",
        "selectable": "true",
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "resourceTimelineDay,resourceTimelineWeek,resourceTimelineMonth",
        },
        "slotMinTime": "06:00:00",
        "slotMaxTime": "18:00:00",
        "initialView": "resourceTimelineDay",
        "resourceGroupField": "Zone",
        "resources": [
            {"id": "a", "zone": "ZONE A", "title": "ZONE A"},
            {"id": "b", "zone": "ZONE A", "title": "ZONE B"},
            {"id": "c", "zone": "ZONE B", "title": "ZONE C"},
            {"id": "d", "zone": "ZONE B", "title": "ZONE D"},
            {"id": "e", "zone": "ZONE C", "title": "ZONE E"},
            {"id": "f", "zone": "ZONE C", "title": "ZONE F"},
        ],
    }

    calendar_events = [
    {
        "title": "Blasting of Zone A",
        "start": "2024-10-10T08:30:00",
        "end": "2024-10-13T10:30:00",
        "resourceId": "a",
    },
    {
        "title": "Drilling of Zone B",
        "start": "2024-10-15T07:30:00",
        "end": "2024-10-16T10:30:00",
        "resourceId": "b",
    },
    {
        "title": "Drilling and blasting of Zone C",
        "start": "2024-10-20T10:40:00",
        "end": "2024-10-21T12:30:00",
        "resourceId": "a",
    }
    ]

    if "resource" in mode:
        if mode == "resource-daygrid":
            calendar_options = {
                **calendar_options,
                "initialDate": "2024-10-10",
                "initialView": "resourceDayGridDay",
                "resourceGroupField": "zone",
            }
        elif mode == "resource-timeline":
            calendar_options = {
                **calendar_options,
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "resourceTimelineDay,resourceTimelineWeek,resourceTimelineMonth",
                },
                "initialDate": "2024-10-10",
                "initialView": "resourceTimelineDay",
                "resourceGroupField": "zone",
            }
        elif mode == "resource-timegrid":
            calendar_options = {
                **calendar_options,
                "initialDate": "2024-10-10",
                "initialView": "resourceTimeGridDay",
                "resourceGroupField": "zone",
            }
    else:
        if mode == "daygrid":
            calendar_options = {
                **calendar_options,
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "dayGridDay,dayGridWeek,dayGridMonth",
                },
                "initialDate": "2024-10-10",
                "initialView": "dayGridMonth",
            }
        elif mode == "timegrid":
            calendar_options = {
                **calendar_options,
                "initialView": "timeGridWeek",
            }
        elif mode == "timeline":
            calendar_options = {
                **calendar_options,
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "timelineDay,timelineWeek,timelineMonth",
                },
                "initialDate": "2024-10-10",
                "initialView": "timelineMonth",
            }
        elif mode == "list":
            calendar_options = {
                **calendar_options,
                "initialDate": "2024-10-10",
                "initialView": "listMonth",
            }
        elif mode == "multimonth":
            calendar_options = {
                **calendar_options,
                "initialView": "multiMonthYear",
            }
    state = calendar(
        events=st.session_state.get("events", calendar_events),
        options=calendar_options,
        custom_css="""
        .fc-event-past {
           opacity: 0.8;
       }
        .fc-event-time {
           font-style: italic;
       }
        .fc-event-title {
           font-weight: 700;
       }
        .fc-toolbar-title {
           font-size: 2rem;
       }
       """,
       key=mode,
    )
    if state.get("eventsSet") is not None:
        st.session_state["events"] = state["eventsSet"]




    #calendar = calendar(events=calendar_events, options=calendar_options, custom_css=custom_css)

    st.write(state)


text = st.text_area(label='Notes', placeholder='Write any notes you would want regarding the blasting operations...', max_chars=1000)