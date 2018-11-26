SELECT * FROM image_info LIMIT 1000;

SELECT 
    imageid, 
    c.classificationname
FROM image_tags a
join tags_classification b on a.imagetagid = b.imagetagid
join classification_info c on c.classificationid = b.classificationid

SELECT * FROM user_info LIMIT 1000;
SELECT * FROM image_info order by imageid DESC Limit 10;
SELECT * FROM image_tags order by imageid DESC Limit 10;
SELECT * FROM tags_classification order by imagetagid DESC limit 20;
SELECT * FROM classification_info LIMIT 1000;
SELECT * FROM image_tagging_state LIMIT 1000;
SELECT * FROM training_info LIMIT 1000;
SELECT * FROM prediction_info LIMIT 1000;

WITH tii AS (
INSERT INTO training_info (trainingdescription, modellocation, createdbyuser) 
VALUES ('This is test training', 'Some URL', 1) returning trainingid),

SELECT classificationid, classificationname FROM classification_info 
where classificationname in ('rattlesnake','goldfinch')

SELECT imagetagid from 

SELECT imagetagid FROM image_tags 
WHERE 
    imageid = 5 and 
    x_min = 232.93 and 
    x_max = 240.68 and 
    y_min = 71.88 and 
    y_max = 278.36
    

INSERT INTO prediction_info (imagetagid,classificationid,trainingid,boxconfidence,imageconfidence)
SELECT imagetagid, classificationid, trainingid
			

order by imageid DESC Limit 10;


INSERT INTO User_Info (UserName) VALUES ('AndreBriggs') 
ON CONFLICT (username) 
DO UPDATE SET username=EXCLUDED.username 
WHERE
    NOT EXISTS (
        SELECT UserId FROM User_Info WHERE UserName = 'AndreBriggs'
    )
RETURNING UserId;


INSERT INTO User_Info (UserName) VALUES ('AndreBriggs3') ;






SELECT 
    imageid, 
    count(*) AS TagCount
FROM image_tags a
--join tags_classification b on a.imagetagid = b.imagetagid
group by imageid order by imageid ASC




--Extract out the file name from image location (do in code)
--Download the images from the image location (to create TF record)
--Currently we don't have db infrastruction for box and image confidence
--Perhaps they are just null (None) for now.
--I'm unclear on what the 'folder' column represents 
    SELECT 
        a.imageid,
        a.imagelocation,
        d.classificationname,
        b.x_min,
        b.x_max,
        b.y_min,
        b.y_max,
        a.height,
        a.width
        --e.username,
    -- b.createddtim
    FROM image_info a join image_tags b on a.imageid = b.imageid
    join tags_classification c on b.imagetagid = c.imagetagid 
    join classification_info d on c.classificationid = d.classificationid 
    join user_info e on b.createdbyuser = e.userid
    WHERE a.height = 50--a.imagelocation LIKE ('%abriglinuxstor%')
    order by d.createddtim desc 
limit 40

--Top Taggers
SELECT 
e.username,
count(*) as TagCount
FROM image_info a join image_tags b on a.imageid = b.imageid
join tags_classification c on b.imagetagid = c.imagetagid 
join classification_info d on c.classificationid = d.classificationid 
join user_info e on b.createdbyuser = e.userid
group by e.username
order by TagCount desc

--Top Taggers By classification_info
SELECT 
e.username,
d.classificationname,
count(*) as TagCount
FROM image_info a join image_tags b on a.imageid = b.imageid
join tags_classification c on b.imagetagid = c.imagetagid 
join classification_info d on c.classificationid = d.classificationid 
join user_info e on b.createdbyuser = e.userid
group by e.username, d.classificationname
order by TagCount desc

-- Top Classifications
SELECT b.classificationname, count(*) AS ClassificationCount FROM tags_classification a
join classification_info b on a.classificationid = b.classificationid
group by classificationname order by ClassificationCount desc

--Tag State by Count
SELECT b.tagstatename, count(*) as Count
FROM image_tagging_state a join tagstate b on a.tagstateid = b.tagstateid
group by tagstatename
order by Count DESC



with iti AS ( 
    INSERT INTO image_tags (ImageId, x_min,x_max,y_min,y_max) 
    VALUES (16, 273.76090362347554,289.7503548737132,274.23149646823157,288.8136207577629) RETURNING ImageTagId), 
    ci AS ( 
        INSERT INTO classification_info (ClassificationName) 
        VALUES ('mackerel'), ('goldfinch'), (' african elephant') 
        ON CONFLICT (ClassificationName) DO UPDATE SET ClassificationName=EXCLUDED.ClassificationName RETURNING (SELECT iti.ImageTagId FROM iti), ClassificationId) 
    INSERT INTO tags_classification (ImageTagId,ClassificationId) 
    SELECT imagetagid,classificationid from ci;

select * from tagstate;

SELECT b.ImageId, b.ImageLocation, a.TagStateId FROM Image_Tagging_State a 
JOIN Image_Info b ON a.ImageId = b.ImageId WHERE a.TagStateId in (2,3) order by 
a.createddtim DESC

    SELECT 
        a.imageid,
        a.imagelocation,
        d.classificationname,
        b.x_min,
        b.x_max,
        b.y_min,
        b.y_max,
        a.height,
        a.width
    FROM image_info a join image_tags b on a.imageid = b.imageid
    join tags_classification c on b.imagetagid = c.imagetagid 
    join classification_info d on c.classificationid = d.classificationid 
    join image_tagging_state e on b.imageid = e.imageid
    WHERE a.height = 50 and e.TagStateId in (2,3) --Cmpleted or in tagging state

