package com.kb.kobo.entity;

import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import javax.persistence.*;
import java.sql.Timestamp;
import java.util.Date;


@Entity
@Table(name="SupportPrograms")
@Getter
@Setter
public class SupportProgram {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "program_id")
    private Long programId;

    @Column(name = "program_name", nullable = false)
    private String programName;

//    @Column(name = "description", nullable = false)
//    private String description;
//
//    @Column(name = "eligibility_criteria", nullable = false)
//    private String eligibilityCriteria;
//
//    @Column(name = "application_deadline", nullable = false)
//    @Temporal(TemporalType.DATE)
//    private Date applicationDeadline;

    @Column(name = "url", nullable = false)
    private String url;

    @CreationTimestamp
    private Timestamp createdAt;

}
