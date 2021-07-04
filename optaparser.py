# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:53:07 2020

@author: ricks
"""
import pathlib
import xml.etree.ElementTree as ET
from typing import List, Union

import numpy as np
import pandas as pd


def add_attacking_direction(eventsDF, tdatDF, playersDBDF, tMetaDF):

    attacking_directions = dict()

    home_gk = playersDBDF.loc[(playersDBDF["position"] == "Goalkeeper")].loc[0][
        "jersey_no"
    ]

    gk_starting_position = tdatDF.loc[
        (tdatDF["frameID"] == tMetaDF["period1_start"])
        & (tdatDF["team"] == 1)
        & (tdatDF["jersey_no"] == int(home_gk))
    ]["x"]

    if int(gk_starting_position) > 0:

        attacking_directions["team1_period1"] = 1
        attacking_directions["team0_period1"] = -1
        attacking_directions["team1_period2"] = -1
        attacking_directions["team0_period2"] = 1

    else:

        attacking_directions["team1_period1"] = -1
        attacking_directions["team0_period1"] = 1
        attacking_directions["team1_period2"] = 1
        attacking_directions["team0_period2"] = -1

    if tMetaDF["period3_end"] != 0:

        home_gk = playersDBDF.loc[(playersDBDF["position"] == "Goalkeeper")].loc[0][
            "jersey_no"
        ]

        gk_starting_position = tdatDF.loc[
            (tdatDF["frameID"] == tMetaDF["period3_start"])
            & (tdatDF["team"] == 1)
            & (tdatDF["jersey_no"] == int(home_gk))
        ]["x"]

        if int(gk_starting_position) > 0:

            attacking_directions["team1_period3"] = 1
            attacking_directions["team0_period3"] = -1
            attacking_directions["team1_period4"] = -1
            attacking_directions["team0_period4"] = 1

        else:

            attacking_directions["team1_period3"] = -1
            attacking_directions["team0_period3"] = 1
            attacking_directions["team1_period4"] = 1
            attacking_directions["team0_period4"] = -1

    else:

        attacking_directions["team1_period3"] = 0
        attacking_directions["team0_period3"] = 0
        attacking_directions["team1_period4"] = 0
        attacking_directions["team0_period4"] = 0

    team_reference = playersDBDF[["team_id", "team"]].drop_duplicates()
    team_reference = team_reference.reset_index(drop=True)

    eventsDF = eventsDF.merge(
        team_reference, left_on="team_id", right_on="team_id", how="outer"
    )

    eventsDF["attacking_direction"] = 0

    for i in range(0, len(eventsDF)):

        ball_to_assess = eventsDF.loc[i]

        if ball_to_assess["period_id"] == 1:

            if ball_to_assess["team"] == 1:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team1_period1"
                ]

            elif ball_to_assess["team"] == 0:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team0_period1"
                ]

        if ball_to_assess["period_id"] == 2:

            if ball_to_assess["team"] == 1:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team1_period2"
                ]

            elif ball_to_assess["team"] == 0:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team0_period2"
                ]

        if ball_to_assess["period_id"] == 3:

            if ball_to_assess["team"] == 1:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team1_period3"
                ]

            elif ball_to_assess["team"] == 0:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team0_period3"
                ]

        if ball_to_assess["period_id"] == 4:

            if ball_to_assess["team"] == 1:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team1_period4"
                ]

            elif ball_to_assess["team"] == 0:
                eventsDF.at[i, "attacking_direction"] = attacking_directions[
                    "team0_period4"
                ]

    return eventsDF


def create_playerDB(f7_filename):

    """
    Function that parses an f7 file to a workable pandas dataframe.

    The input is an xml that contains valuable playerinformation.

    Different steps are taken to get to the final database.
    These steps are based on the strange and weird structure of the xml file which was hard to work with.

    1. Create a dataframe of all strating players. This contains:
        - Formation place
        - playerID
        - Position
        - shirt number
        - status (Starter or substitute)
    2. create a dataframe of all player info. This contains:
        - playerID
        - First_name
        - last_name
        - full_name
    3. merge both dataframes
    4. Add playing time and work around substitutions. This contains:
        - full time in minutes
        - minutes seperately
        - seconds seperately
    5. Add teamID for each player (THIS WAS TRICKY, NEED TO CHECK FOR MULTIPLE MATCHES)

    output is dataframe containing the above
    """

    tree = ET.parse(f7_filename)
    root = tree.getroot()

    # match_id = int(root.find("SoccerDocument").get("uID")[1:])

    gameinfo = root.findall("SoccerDocument")[0]
    # gameinfo = gameinfo_1[0]

    formation_place = []
    player_id = []
    position = []
    jersey_no = []
    status = []

    for neighbor in gameinfo.iter("MatchPlayer"):
        formation_place.append(neighbor.get("Formation_Place"))
        player_id.append(neighbor.get("PlayerRef")[1:])
        position.append(neighbor.get("Position"))
        jersey_no.append(neighbor.get("ShirtNumber"))
        status.append(neighbor.get("Status"))

    starting_players = pd.DataFrame(
        {
            "formation_place": formation_place,
            "player_id": player_id,
            "position": position,
            "jersey_no": jersey_no,
            "status": status,
        }
    )

    p_id = []
    first_name = []
    last_name = []
    player_name = []

    for neighbor in gameinfo.iter("Player"):
        p_id.append(neighbor.get("uID")[1:])
        first_name.append(neighbor.find("PersonName").find("First").text)
        last_name.append(neighbor.find("PersonName").find("Last").text)
        player_name.append(first_name[-1] + " " + last_name[-1])

    bench_players = pd.DataFrame(
        {
            "first_name": first_name,
            "player_id": p_id,
            "last_name": last_name,
            "player_name": player_name,
        }
    )

    players = starting_players.merge(bench_players, on="player_id", how="inner")

    time = []
    period_id = []
    player_off = []
    player_on = []

    for neighbor in gameinfo.iter("Substitution"):
        time.append(int(neighbor.get("Min")) + int(neighbor.get("Sec")) / 60)
        period_id.append(neighbor.get("Period"))
        player_off.append(neighbor.get("SubOff")[1:])
        if not neighbor.get("Retired") == '1':
            player_on.append(neighbor.get("SubOn")[1:])
        else:
            player_on.append('None')
    subs = pd.DataFrame(
        {
            "time": time,
            "period_id": period_id,
            "player_off": player_off,
            "player_on": player_on,
        }
    )

    players["start_min"] = 0
    players["end_min"] = 0

    for neighbor in gameinfo.iter("Stat"):
        if neighbor.get("Type") == "match_time":
            match_length = int(neighbor.text)

    players.loc[players["status"] == "Start", "end_min"] = match_length

    for index, content in subs.iterrows():
        players.loc[players["player_id"] == content["player_off"], "end_min"] = content[
            "time"
        ]
        players.loc[
            players["player_id"] == content["player_on"], "start_min"
        ] = content["time"]
        players.loc[
            players["player_id"] == content["player_on"], "end_min"
        ] = match_length

    for neighbor in gameinfo.iter("Booking"):
        if neighbor.get("Card") == "Red":
            players.loc[
                players["player_id"] == neighbor.get("PlayerRef")[1:], "end_min"
            ] = (int(neighbor.get("Min")) + int(neighbor.get("Sec")) / 60)

    players["mins_played"] = players["end_min"] - players["start_min"]

    # players["match_id"] = match_id

    home_away = []

    for team in gameinfo.findall("Team"):
        home_away.append(team.get("uID")[1:])

    players = players[players.mins_played != 0]
    players = players.reset_index(drop=True)

    subs_index = players[(players.status == "Sub")].index
    diff_subs_index = np.diff(subs_index)

    for i in range(len(subs_index) - 1):
        if subs_index[i + 1] - subs_index[i] > 1:
            index_finder = subs_index[i] + 1
            players.loc[:index_finder, "team"] = home_away[0]
            players.loc[index_finder:, "team"] = home_away[1]

    if np.max(diff_subs_index) == 1:
        if np.min(subs_index) < 20:
            index_finder = 11
            players.loc[:index_finder, "team"] = home_away[0]
            players.loc[index_finder:, "team"] = home_away[1]

        else:
            index_finder = 11
            players.loc[:index_finder, "team"] = home_away[0]
            players.loc[index_finder:, "team"] = home_away[1]

    return players, home_away


def parse_f24(file_name):

    # parse the xml and convert to a tree and root
    tree = ET.parse(file_name)
    root = tree.getroot()

    # get the main game info from the single 'Game' node
    gameinfo = root.findall("Game")
    gameinfo = gameinfo[0]
    game_id = gameinfo.get("id")
    home_team_id = gameinfo.get("home_team_id")
    home_team_name = gameinfo.get("home_team_name")
    away_team_id = gameinfo.get("away_team_id")
    away_team_name = gameinfo.get("away_team_name")
    competition_id = gameinfo.get("competition_id")
    competition_name = gameinfo.get("competition_name")
    season_id = gameinfo.get("season_id")

    Edata_columns = [
        "id",
        "event_id",
        "type_id",
        "period_id",
        "min",
        "sec",
        "outcome",
        "player_id",
        "team_id",
        "x",
        "y",
        "sequence_id",
        "possession_id",
    ]

    Q_ids = []
    Q_values = []
    Edata = []

    # loop through each ball node and grab the information
    for i in root.iter("Event"):

        # get the info from the ball node main chunk
        id_ = int(i.get("id"))
        event_id = i.get("event_id")
        type_id = i.get("type_id")
        period_id = int(i.get("period_id"))
        outcome = int(i.get("outcome"))
        min_ = int(i.get("min"))
        sec = int(i.get("sec"))
        player_id = i.get("player_id")
        team_id = i.get("team_id")
        x = i.get("x")
        y = i.get("y")
        possession_id = i.get("possession_id")
        sequence_id = i.get("sequence_id")

        Edata_values = [
            id_,
            event_id,
            type_id,
            period_id,
            min_,
            sec,
            outcome,
            player_id,
            team_id,
            x,
            y,
            sequence_id,
            possession_id,
        ]

        # find all of the Q information for that file
        Qs = i.findall("./Q")

        # create some empty lists to append the results to
        qualifier_id = []
        Q_value = []

        # loop through all of the Qs and grab the info
        for child in Qs:
            qualifier_id.append(child.get("qualifier_id"))
            Q_value.append(child.get("value", default="1"))

        Q_ids.append(qualifier_id)
        Q_values.append(Q_value)
        Edata.append(Edata_values)

    # Stack all ball Data
    df = pd.DataFrame(np.vstack(Edata), columns=Edata_columns)

    unique_Q_ids = np.unique(np.concatenate(Q_ids))

    # create an array for fast assignments
    Qarray = np.zeros((df.shape[0], len(unique_Q_ids)))
    Qarray = Qarray.astype("O")
    Qarray[:] = np.nan

    # dict to relate Q_ids to array indices
    keydict = dict(zip(unique_Q_ids, range(len(unique_Q_ids))))

    # iter through all Q_ids, Q_values, assign values to appropriate indices
    for idx, (i, v) in enumerate(zip(Q_ids, Q_values)):
        Qarray[idx, [keydict.get(q) for q in Q_ids[idx]]] = Q_values[idx]

    # df from array
    Qdf = pd.DataFrame(Qarray, columns=unique_Q_ids, index=df.index)

    # combine
    game_df = pd.concat([df, Qdf], axis=1)

    # assign game values
    game_df["competition_id"] = competition_id
    game_df["game_id"] = game_id
    game_df["home_team_id"] = home_team_id
    game_df["home_team_name"] = home_team_name
    game_df["away_team_id"] = away_team_id
    game_df["away_team_name"] = away_team_name
    game_df["competition_id"] = competition_id
    game_df["competition_name"] = competition_name
    game_df["season_id"] = season_id
    game_df["competition_id"] = competition_id

    game_df[["id", "period_id", "min", "sec", "outcome", "140", "141"]] = game_df[
        ["id", "period_id", "min", "sec", "outcome", "140", "141"]
    ].astype("float")

    game_df["x"] = pd.to_numeric(game_df["x"])
    game_df["y"] = pd.to_numeric(game_df["y"])

    game_df[['x', '140']] = game_df[['x', '140']] / 100 * 105
    game_df[['y', '141']] = game_df[['y', '141']] / 100 * 68
    for i in root.iter("Game"):
        play_date = i.get("game_date").split('T')[0]   

    # write to csv
    return game_df, play_date


def parse_tracab(
    tracking_filename: Union[str, pathlib.Path],
    game_metadata: pd.DataFrame,
    home_away: List,
) -> pd.DataFrame:

    """
    Parse a tracab.dat file and convert it to a workable pandas dataframe.

    Tracking_filename: File containing tracking data of all players and the ball
        File contains:
            - FrameID = captured frame of datapoints
            - team = 1: home and 0: away 10: ball
            - target_id = set player to a nummeric value (make data anonymous). Value of 100 for ball.
            - jersey_nu = players jersey number, 999 for ball
            - x = position of player/ball along the length of the pitch in meters
            - y = position of player/ball olong the width of the pitch in meters
            - z = height of the ball in meters
            - owning_team = team in possession of the ball A for away and H for Home
            - ball_status = ball in or out of play: Alive = in play and Dead = ball out of play
            - Ball_contact = value for the ball not sure what is here. B4: when ball is dead;
              Whistle, SetHome, SetAway

    Metadata_filename: File containing
        File contains:
            - Values corresponding to the starting frame of the first and second half
            - Values corresponding to the ending frame of the first and second half
            - pitch length and pitch width in meters

    After parsing all the data based on the settings below the dataframe can be cleaned.
    Set values for removing officials and tream_dead_time
    removing officials = true -> remove officials; false = don't remove
    trim_dead_time = True -> keep data when ball is out of play; false = delete data when ball is out of play

    Output: Dataframe of the entire match of approximately 3.3 milion datapoints and 10 columns
    """

    remove_officials = True
    trim_dead_time = True

    # First save all the dataframes/lines from the .dat file in a list
    # This will take a lot of time initially

    with open(tracking_filename) as fn:
        file_content = fn.readlines()

    content_raw = [x.strip() for x in file_content]

    # Create empty lists to store the data per variable
    frameID = []
    team = []
    target_id = []
    jersey_no = []
    x = []
    y = []
    z = []
    speed = []
    ball_owning_team = []
    ball_status = []

    for data_row in content_raw:

        data_split = data_row.replace(":", ";").split(";")
        data_split = list(filter(None, data_split))

        ball_data_split = data_split[-1].split(",")

        frameID.append(int(data_split[0]))
        team.append("10")
        target_id.append("100")
        jersey_no.append("999")
        x.append(float(ball_data_split[0]))
        y.append(float(ball_data_split[1]))
        z.append(float(ball_data_split[2]))
        speed.append(float(ball_data_split[3]))
        ball_owning_team.append(ball_data_split[4])
        ball_status.append(ball_data_split[5])

        for content in data_split[1:-1]:
            split_content = content.split(",")
            frameID.append(int(data_split[0]))
            team.append(split_content[0])
            target_id.append(split_content[1])
            jersey_no.append(split_content[2])
            x.append(float(split_content[3]))
            y.append(float(split_content[4]))
            z.append(float(0.0))
            speed.append(float(split_content[5]))
            ball_owning_team.append(ball_data_split[4])
            ball_status.append(ball_data_split[5])

    all_tracking_data = pd.DataFrame(
        {
            "frameID": frameID,
            "team": team,
            "target_id": target_id,
            "jersey_no": jersey_no,
            "x": x,
            "y": y,
            "z": z,
            "speed": speed,
            "ball_owning_team": ball_owning_team,
            "ball_status": ball_status,
        }
    )

    # convert x, y and z to meters and set (0,0) at the bottom left.
    all_tracking_data["x"] = (
        all_tracking_data["x"] / 100 + game_metadata.loc[0, "pitch_x"] / 2
    )
    all_tracking_data["y"] = (
        all_tracking_data["y"] / 100 + game_metadata.loc[0, "pitch_y"] / 2
    )
    all_tracking_data["z"] = all_tracking_data["z"] / 100

    if remove_officials:
        use = ["1", "0", "10"]
        all_tracking_data = all_tracking_data[all_tracking_data.team.isin(use)]

    if trim_dead_time:
        if game_metadata.loc[0, "period3_start"] == 0:
            all_tracking_data = all_tracking_data[
                (
                    (
                        all_tracking_data["frameID"]
                        >= game_metadata.loc[0, "period1_start"]
                    )
                    & (
                        all_tracking_data["frameID"]
                        <= game_metadata.loc[0, "period1_end"]
                    )
                )
                | (
                    (
                        all_tracking_data["frameID"]
                        >= game_metadata.loc[0, "period2_start"]
                    )
                    & (
                        all_tracking_data["frameID"]
                        <= game_metadata.loc[0, "period2_end"]
                    )
                )
            ]

    all_tracking_data = all_tracking_data.reset_index(drop=True)

    all_tracking_data.loc[all_tracking_data.loc[:, "team"] == "0", "team"] = home_away[
        1
    ]
    all_tracking_data.loc[all_tracking_data.loc[:, "team"] == "1", "team"] = home_away[
        0
    ]

    return all_tracking_data


def parse_tracking_metadata(
    metadata_filename: Union[str, pathlib.Path]
) -> pd.DataFrame:
    """
    An xml file will be parsed.

    Output = Dataframe containing:
        - FrameID of start and end of first and second half
        - Length and width of the pitch

    Remarks: period 3 and 4 have values of 0 if there was no overtime. Overtime can only happen in cupmatches.
    """
    tree = ET.parse(metadata_filename)
    root = tree.getroot()

    # period_startframe = []
    # period_endframe = []

    gamexml = root.findall("match")[0]
    # gamexml.findall('period').get('iStartFrame')

    info_raw = []

    for i in gamexml.iter("period"):
        # get the info from the ball node main chunk
        #         print(int(i.get('iId')))
        info_raw.append(i.get("iStartFrame"))
        info_raw.append(i.get("iEndFrame"))

    # # Create empty dict Capitals
    game_metadata = pd.DataFrame()

    # # Fill it with some values
    game_metadata.loc[0, "period1_start"] = pd.to_numeric(info_raw[0])
    game_metadata.loc[0, "period1_end"] = pd.to_numeric(info_raw[1])
    game_metadata.loc[0, "period2_start"] = pd.to_numeric(info_raw[2])
    game_metadata.loc[0, "period2_end"] = pd.to_numeric(info_raw[3])
    game_metadata.loc[0, "period3_start"] = pd.to_numeric(info_raw[4])
    game_metadata.loc[0, "period3_end"] = pd.to_numeric(info_raw[5])
    game_metadata.loc[0, "period4_start"] = pd.to_numeric(info_raw[6])
    game_metadata.loc[0, "period4_end"] = pd.to_numeric(info_raw[7])

    for detail in root.iter("match"):
        game_metadata.loc[0, "pitch_x"] = pd.to_numeric(detail.get("fPitchXSizeMeters"))
        game_metadata.loc[0, "pitch_y"] = pd.to_numeric(detail.get("fPitchYSizeMeters"))

    return game_metadata
